import os
import cv2
import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torchvision.models import swin_t, Swin_T_Weights

#############################################
# Dataset that reads from a zip file        #
#############################################

import os
import zipfile
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class NYUDataset(Dataset):
    def __init__(self, zip_path, csv_path_within_zip, base_dir_within_zip, transform=None, max_rows=5945):
        self.zip_path = zip_path
        self.csv_path_within_zip = csv_path_within_zip
        self.base_dir = base_dir_within_zip  # e.g., "nyu_data/data"
        self.transform = transform
        self.max_rows = 13768  # Store the maximum number of rows

        with zipfile.ZipFile(self.zip_path, 'r') as z:
            with z.open(self.csv_path_within_zip) as f:
                self.data_info = pd.read_csv(f, header=None).iloc[:self.max_rows]
                
        self.zip_file = None

    def _open_zip(self):
        if self.zip_file is None:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
        return self.zip_file

    def __len__(self):
        return len(self.data_info)  # This will return max_rows (e.g., 5945) or less if CSV is smaller

    def __getitem__(self, idx):
        img_rel_path = self.data_info.iloc[idx, 0].strip()    # e.g., "data/nyu2_train/living_room_0038_out/37.jpg"
        depth_rel_path = self.data_info.iloc[idx, 1].strip()  # e.g., "data/nyu2_train/living_room_0038_out/37.png"

        if img_rel_path.startswith("data/"):
            img_rel_path = img_rel_path[len("data/"):]
        if depth_rel_path.startswith("data/"):
            depth_rel_path = depth_rel_path[len("data/"):]
        
        img_zip_path = os.path.join(self.base_dir, img_rel_path).replace("\\", "/")
        depth_zip_path = os.path.join(self.base_dir, depth_rel_path).replace("\\", "/")
        
        z = self._open_zip()
        
        # Read and decode the RGB image.
        with z.open(img_zip_path) as f:
            img_bytes = f.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found or failed to decode: {img_zip_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read and decode the depth image.
        with z.open(depth_zip_path) as f:
            depth_bytes = f.read()
        depth_array = np.frombuffer(depth_bytes, np.uint8)
        depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth file not found or failed to decode: {depth_zip_path}")

        # Do NOT resize: use original image sizes (expected: height=480, width=640)
        # Normalize both image and depth: image already scaled by 255.0; now scale depth to [0,1]
        image = image.astype(np.float32) / 255.0
        depth = depth.astype(np.float32) / 255.0
        
        # Convert image to tensor with shape (3, 480, 640)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        
        # Ensure depth tensor shape is (1, 480, 640)
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=0)
        else:
            depth = np.transpose(depth, (2, 0, 1))
        depth = torch.from_numpy(depth)
        
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample
    

class DepthEstimationSwin(nn.Module):
    def __init__(self, img_height=480, img_width=640):
        super(DepthEstimationSwin, self).__init__()
        
        # Load pretrained Swin-T backbone
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone.head = nn.Identity()  # Remove classification head
        
        # Decoder to upsample from [B, 768, 15, 20] to [B, 1, 480, 640]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),  # [B, 256, 30, 40]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # [B, 128, 60, 80]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # [B, 64, 120, 160]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # [B, 32, 240, 320]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2),     # [B, 1, 480, 640]
            nn.Sigmoid()  # Normalize depth to [0, 1]
        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.backbone.features(x)  # Output: [B, 300, 768]
        x = self.backbone.norm(x)  # Output: [B, 300, 768]
        x = x.view(B, 15, 20, 768).permute(0, 3, 1, 2)  # Output: [B, 768, 15, 20]
        depth_map = self.decoder(x)  # Output: [B, 1, 480, 640]
        
        return depth_map

def visualize_prediction(rgb_tensor, depth_gt_tensor, depth_pred_tensor):
    """
    Visualizes the RGB image, ground truth depth, and predicted depth.
    """
    # Convert RGB tensor (3, H, W) to numpy (H, W, 3)
    rgb_np = rgb_tensor.cpu().numpy().transpose(1, 2, 0)
    # Squeeze depth tensors to (H, W)
    depth_gt_np = depth_gt_tensor.cpu().numpy().squeeze()
    depth_pred_np = depth_pred_tensor.cpu().numpy().squeeze()
    
    # Ensure the displayed depth is within [0,1]
    depth_gt_np = np.clip(depth_gt_np, 0, 1)
    depth_pred_np = np.clip(depth_pred_np, 0, 1)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_np, vmin=0, vmax=1)
    plt.title("RGB Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(depth_gt_np, cmap='inferno', vmin=0, vmax=1)
    plt.title("Ground Truth Depth")
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(depth_pred_np, cmap='inferno', vmin=0, vmax=1)
    plt.title("Predicted Depth")
    plt.axis('off')
    plt.colorbar()
    
    plt.show()

def berhu_loss(pred, target, mask=None):
    """
    Compute the BerHu (reverse Huber) loss.
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = torch.abs(pred - target)
    c = 0.2 * diff.max().item()
    loss = torch.where(diff <= c, diff, (diff**2 + c**2) / (2 * c))
    return loss.mean()

def edge_aware_smoothness_loss(pred, image):
    """
    Compute an edge-aware smoothness loss on the predicted depth map.
    """
    grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    grad_img_x = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    weight_x = torch.exp(-grad_img_x)
    weight_y = torch.exp(-grad_img_y)
    loss_x = grad_pred_x * weight_x
    loss_y = grad_pred_y * weight_y
    return torch.mean(loss_x) + torch.mean(loss_y)

def depth_loss(pred, target, image, lambda_smooth=0.04):
    """
    Combined loss function for monocular depth estimation.
    """
    mse_loss = nn.MSELoss()  # Instantiate the MSE loss function
    loss_depth = berhu_loss(pred, target)
    loss_smooth = edge_aware_smoothness_loss(pred, image)
    loss_mse = mse_loss(pred, target)  # Compute MSE
    return loss_depth + lambda_smooth * loss_smooth + loss_mse * 0.2


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # The zip file "archive.zip" should contain "nyu_data" folder with "data" inside it,
    # which in turn contains "nyu2_train" folder and "nyu2_train.csv".
    zip_path = "archive.zip"
    csv_path_within_zip = "nyu_data/data/nyu2_train.csv"
    base_dir_within_zip = "nyu_data/data"

    # Instantiate the dataset
    dataset = NYUDataset(
        zip_path=zip_path,
        csv_path_within_zip=csv_path_within_zip,
        base_dir_within_zip=base_dir_within_zip,
        transform=None,
        max_rows=13768    # Limit to 5945 rows
    )
    train_loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4)

    model = DepthEstimationSwin().to(device)
    num_epochs = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # checkpoint = torch.load('Model6840.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # #model.load_state_dict(torch.load('new_model400.pth'))
    i = 6300
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            images = batch['image'].to(device)   # [B, 3, 480, 640]
            depths = batch['depth'].to(device)     # [B, 1, 480, 640]
            optimizer.zero_grad()
            preds = model(images)                  # [B, 1, 480, 640]
            loss = depth_loss(preds, depths, images, lambda_smooth=0.1)
            loss.backward()
            optimizer.step()
            i += 1

            if i % 40 == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'Model{i}.pth')
            elif i % 10 == 0:
                print(f" Epoch {i}, Loss: {loss.item():.4f}")

    # Uncomment below to visualize a prediction from one sample.
    sample = next(iter(train_loader))
    rgb_sample = sample['image'][0].to(device)
    depth_gt_sample = sample['depth'][0].to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        depth_pred_sample = model(rgb_sample.unsqueeze(0))[0]
    visualize_prediction(rgb_sample, depth_gt_sample, depth_pred_sample)
