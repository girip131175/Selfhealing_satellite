import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import random
import time
import torch.amp

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = True

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
dataset_path = os.environ.get("SELF_HEALING_EDGE_DATASET", os.path.join(os.path.dirname(__file__), "data", "pix2pix_dataset"))
normal_dir = os.path.join(dataset_path, "normal")
damaged_dir = os.path.join(dataset_path, "damaged")
output_dir = os.path.join(dataset_path, "results")
checkpoint_dir = os.path.join(output_dir, "checkpoints")
sample_dir = os.path.join(output_dir, "samples")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

# Hyperparameters
epochs = 50
batch_size = 16
learning_rate = 0.0002
beta1, beta2 = 0.5, 0.999
lambda_L1 = 100
image_size = 256

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset Class
class SatelliteImageDataset(Dataset):
    def _init_(self, normal_dir, damaged_dir, transform=None):
        self.transform = transform
        self.paired_images = []
        self.cache = {}

        normal_images = {img: os.path.join(root, img)
                        for root, _, files in os.walk(normal_dir)
                        for img in files if img.lower().endswith(('.png', '.jpg', '.jpeg'))}

        damaged_images = {img: os.path.join(root, img)
                         for root, _, files in os.walk(damaged_dir)
                         for img in files if img.lower().endswith(('.png', '.jpg', '.jpeg'))}

        self.paired_images = [(normal_images[img], damaged_images[img])
                            for img in normal_images.keys() if img in damaged_images]

        if not self.paired_images:
            raise ValueError("No paired images found. Check dataset structure.")
        print(f"Total paired images: {len(self.paired_images)}")

    def _len_(self):
        return len(self.paired_images)

    def _getitem_(self, idx):
        normal_path, damaged_path = self.paired_images[idx]
        if idx not in self.cache:
            normal_image = Image.open(normal_path).convert("RGB")
            damaged_image = Image.open(damaged_path).convert("RGB")
            if self.transform:
                normal_image = self.transform(normal_image)
                damaged_image = self.transform(damaged_image)
            self.cache[idx] = (normal_image, damaged_image)
        return self.cache[idx]

# UNet Generator
class UNetGenerator(nn.Module):
    def _init_(self):
        super(UNetGenerator, self)._init_()
        
        def down_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        def up_conv(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
        
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc2 = down_conv(64, 128)
        self.enc3 = down_conv(128, 256)
        self.enc4 = down_conv(256, 512)
        self.dec1 = up_conv(512, 256)
        self.dec2 = up_conv(512, 128)
        self.dec3 = up_conv(256, 64)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d1 = self.dec1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        return self.dec4(d3)

# PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    def _init_(self):
        super(PatchDiscriminator, self)._init_()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Training Function
def train(generator, discriminator, train_loader, optimizer_G, optimizer_D):
    generator.train()
    discriminator.train()
    scaler = torch.amp.GradScaler('cuda')
    
    print("Model loaded and training started")
    
    for epoch in range(epochs):
        start_time = time.time()
        g_loss_total = 0.0
        d_loss_total = 0.0
        num_batches = 0
        
        for normal_images, damaged_images in train_loader:
            normal_images = normal_images.to(device)
            damaged_images = damaged_images.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            with torch.amp.autocast('cuda'):
                fake_images = generator(normal_images)
                real_validity = discriminator(damaged_images)
                fake_validity = discriminator(fake_images.detach())
                d_loss = criterion_GAN(real_validity, torch.ones_like(real_validity)) + \
                        criterion_GAN(fake_validity, torch.zeros_like(fake_validity))
                d_loss = d_loss / 2
            
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            
            # Train Generator
            optimizer_G.zero_grad()
            with torch.amp.autocast('cuda'):
                fake_images = generator(normal_images)
                fake_validity = discriminator(fake_images)
                g_loss = criterion_GAN(fake_validity, torch.ones_like(fake_validity)) + \
                        lambda_L1 * criterion_L1(fake_images, damaged_images)
            
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()
            
            # Accumulate losses
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            num_batches += 1
        
        epoch_time = time.time() - start_time
        avg_g_loss = g_loss_total / num_batches
        avg_d_loss = d_loss_total / num_batches
        
        print(f"Time taken for epoch {epoch+1} = {epoch_time:.2f}s, G_loss = {avg_g_loss:.4f}, D_loss = {avg_d_loss:.4f}")
        
        # Save checkpoint and samples
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        with torch.no_grad():
            fake_images = generator(normal_images[:4])
            save_image(fake_images, os.path.join(sample_dir, f"epoch_{epoch+1}.png"), 
                      nrow=4, normalize=True)

# Main execution
try:
    dataset = SatelliteImageDataset(normal_dir, damaged_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, drop_last=True)
    
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    
    train(generator, discriminator, train_loader, optimizer_G, optimizer_D)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")

print("Training completed!")
