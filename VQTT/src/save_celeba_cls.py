import torch
from torch import nn
from transformers import ViTImageProcessor, ViTModel
import os
from PIL import Image
import requests
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class DinoExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        self.model = nn.Sequential(
            torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'),
        )
        
        self.model = self.model.to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, images):
        images = images.to(self.device)
        out = self.model(images)
        return out.cpu()


class CelebA(Dataset):
    def __init__(self, img_dir="/home/shared/data/celeba/img_align_celeba", 
                 transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.filenames = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # image = transforms.ToTensor()(image)
        return image, idx
    
    
def compute_and_save_features(
    img_dir,
    batch_size=32,
    output_file="celeba_dino_cls.pt",
    device="cuda"
):
    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = CelebA(img_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = DinoExtractor(device=device)

    num_images = len(dataset)
    feature_dim = 384  # CLS size for DINO-S/16
    features = torch.zeros(num_images, feature_dim)

    with torch.no_grad():
        for images, idxs in tqdm(loader, total=len(loader), desc="Extracting DINO features"):
            cls_batch = model(images)  # (B, 384)
            features[idxs] = cls_batch

    torch.save(features, output_file)
    print(f"Saved CLS features to: {output_file}")
    

if __name__ == "__main__":
    compute_and_save_features(
        img_dir="/home/shared/data/celeba/img_align_celeba",
        output_file="/home/shared/data/celeba_cls/celeba_dino_cls.pt",
        batch_size=512,
    )
