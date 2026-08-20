import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance
import os
import logging
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UNet Generator class with residual connections
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        def down_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        def up_conv(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
        
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc1_res = nn.Conv2d(64, 64, 3, 1, 1)  # Residual connection
        self.enc2 = down_conv(64, 128)
        self.enc2_res = nn.Conv2d(128, 128, 3, 1, 1)
        self.enc3 = down_conv(128, 256)
        self.enc3_res = nn.Conv2d(256, 256, 3, 1, 1)
        self.enc4 = down_conv(256, 512)
        self.enc4_res = nn.Conv2d(512, 512, 3, 1, 1)
        
        # Decoder
        self.dec1 = up_conv(512, 256)
        self.dec2 = up_conv(512, 128)  # 256 (from dec1) + 256 (from enc3)
        self.dec3 = up_conv(256, 64)   # 128 (from dec2) + 128 (from enc2)
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e1 = e1 + self.enc1_res(e1)  # Residual connection
        e2 = self.enc2(e1)
        e2 = e2 + self.enc2_res(e2)
        e3 = self.enc3(e2)
        e3 = e3 + self.enc3_res(e3)
        e4 = self.enc4(e3)
        e4 = e4 + self.enc4_res(e4)
        
        d1 = self.dec1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        return self.dec4(d3)

# Function to load and preprocess image
def load_image(image_path, transform):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to post-process the reconstructed image
def post_process_image(image_np):
    # Convert to PIL for processing
    image = Image.fromarray((image_np * 255).astype(np.uint8))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Apply sharpening
    image = image.filter(ImageFilter.SHARPEN)
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Convert to grayscale using luminance (weighted average)
    image_np = (0.2989 * image_np[:, :, 0] + 0.5870 * image_np[:, :, 1] + 0.1140 * image_np[:, :, 2])[..., np.newaxis]
    image_np = np.repeat(image_np, 3, axis=2)
    
    return image_np

# Function to detect damage using SSIM and edge detection
def is_damaged(input_tensor, reconstructed_tensor, ssim_threshold=0.5, edge_threshold=0.1):
    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    recon_np = reconstructed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    input_np = (input_np * 0.5) + 0.5  # Denormalize to [0, 1]
    recon_np = (recon_np * 0.5) + 0.5
    
    # Compute SSIM
    similarity = ssim(input_np, recon_np, channel_axis=-1, data_range=1.0)
    logger.info(f"SSIM: {similarity:.4f}")
    
    # Edge detection to check for damage (high edge density indicates damage)
    input_gray = np.mean(input_np, axis=2)
    edges = canny(input_gray, sigma=1.0)
    edge_density = np.mean(edges)
    logger.info(f"Edge density: {edge_density:.4f}")
    
    # Classify as damaged if SSIM is low OR edge density is high
    is_damaged_ssim = similarity < ssim_threshold
    is_damaged_edges = edge_density > edge_threshold
    return is_damaged_ssim or is_damaged_edges

# Function to save input and reconstructed images side by side
def save_input_and_reconstructed(input_tensor, reconstructed_tensor, output_path):
    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    recon_np = reconstructed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    input_np = (input_np * 0.5) + 0.5  # Denormalize to [0, 1]
    recon_np_raw = recon_np
    recon_np = (recon_np * 0.5) + 0.5
    
    # Debug: Log the range of raw and denormalized reconstructed tensor
    logger.info(f"Raw reconstructed tensor range: min={recon_np_raw.min():.4f}, max={recon_np_raw.max():.4f}")
    logger.info(f"Denormalized reconstructed tensor range: min={recon_np.min():.4f}, max={recon_np.max():.4f}")
    
    # Post-process the reconstructed image
    recon_np = post_process_image(recon_np)
    
    input_np = np.clip(input_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)
    combined = np.hstack([input_np, recon_np])
    combined_image = Image.fromarray((combined * 255).astype(np.uint8))
    combined_image.save(output_path)
    logger.info(f"Input and reconstructed images saved at {output_path}")

# Function to load state dictionary with key mismatch handling
def load_model_with_key_check(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle nested checkpoint from edge_connect_gan.py
    if 'generator' in checkpoint:
        checkpoint = checkpoint['generator']
    
    model_dict = model.state_dict()
    checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    missing_keys = [k for k in model_dict.keys() if k not in checkpoint]
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys}")
    
    model_dict.update(checkpoint)
    model.load_state_dict(model_dict, strict=False)
    logger.info("Model state dictionary loaded with flexible key matching")
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    generator = UNetGenerator().to(device)
    checkpoint_path = os.environ.get("SELF_HEALING_GAN_CHECKPOINT", os.path.join(os.path.dirname(__file__), "models", "generator_epoch_50.pth"))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    generator = load_model_with_key_check(generator, checkpoint_path, device)
    generator.eval()
    logger.info("Loaded trained generator model")

    max_attempts = 3
    for attempt in range(max_attempts):
        image_path = input("Enter the path to the satellite image (.png): ").strip()
        image_path = os.path.normpath(image_path.strip('r"').strip('"').strip("'"))
        
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found at {image_path}. Please check the path and try again.")
                if attempt < max_attempts - 1:
                    logger.info(f"Attempts remaining: {max_attempts - attempt - 1}")
                    continue
                else:
                    raise FileNotFoundError(f"Image not found after {max_attempts} attempts.")
            
            input_image = load_image(image_path, transform).to(device)
            
            with torch.no_grad():
                reconstructed_image = generator(input_image)
            
            output_dir = os.path.dirname(image_path)
            comparison_filename = f"comparison_{os.path.basename(image_path)}"
            comparison_path = os.path.join(output_dir, comparison_filename)
            save_input_and_reconstructed(input_image, reconstructed_image, comparison_path)
            
            if is_damaged(input_image, reconstructed_image):
                logger.info("Damage detected in the input image")
                output_filename = f"reconstructed_{os.path.basename(image_path)}"
                output_path = os.path.join(output_dir, output_filename)
                save_input_and_reconstructed(input_image, reconstructed_image, output_path)
            else:
                logger.info("No significant damage detected in the input image")
            
            break
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            if attempt < max_attempts - 1:
                logger.info(f"Attempts remaining: {max_attempts - attempt - 1}")
                continue
            else:
                raise Exception(f"Failed to process image after {max_attempts} attempts: {e}")

if __name__ == "__main__":
    main()
