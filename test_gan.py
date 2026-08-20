import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# UNet Generator (same architecture as in training)
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

def load_image(image_path, transform):
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def save_image(tensor, output_path):
    """Save tensor as image"""
    # Denormalize the tensor
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(tensor)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Reconstructed image saved to: {output_path}")

def reconstruct_image(image_path, model_path, output_dir):
    """Reconstruct an image using the trained GAN model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load the image
    input_tensor = load_image(image_path, transform)
    if input_tensor is None:
        return
    
    # Load the model
    try:
        generator = UNetGenerator().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate reconstructed image
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        reconstructed = generator(input_tensor)
    
    # Save the input and reconstructed images
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_save_path = os.path.join(output_dir, f"{image_name}_input.png")
    output_save_path = os.path.join(output_dir, f"{image_name}_reconstructed.png")
    
    save_image(input_tensor, input_save_path)
    save_image(reconstructed, output_save_path)
    
    # Display comparison
    print(f"Input image saved to: {input_save_path}")
    print(f"Reconstructed image saved to: {output_save_path}")
    
    return input_save_path, output_save_path

def main():
    parser = argparse.ArgumentParser(description="Test GAN model for image reconstruction")
    parser.add_argument('--image_path', type=str, help="Path to the input image")
    parser.add_argument('--model_path', type=str, default=os.environ.get("SELF_HEALING_GAN_CHECKPOINT", os.path.join(os.path.dirname(__file__), "models", "generator_epoch_50.pth")), 
                        help="Path to the trained model checkpoint")
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), "outputs"), 
                        help="Directory to save the reconstructed image")
    
    args = parser.parse_args()
    
    # If image_path is not provided, prompt the user
    if not args.image_path:
        while True:
            user_input = input("Enter the path of the image to reconstruct (must be a .png, .jpg, or .jpeg file): ").strip()
            # Remove quotes if present
            if user_input.startswith('"') and user_input.endswith('"'):
                user_input = user_input[1:-1]
            
            # Normalize path separators for Windows
            user_input = os.path.normpath(user_input)
            
            # Validate the path
            if not os.path.exists(user_input):
                print("Error: The specified file does not exist. Please provide a valid path.")
                continue
            if not user_input.lower().endswith(('.png', '.jpg', '.jpeg')):
                print("Error: The file must be an image (.png, .jpg, or .jpeg). Please provide a valid image file.")
                continue
            args.image_path = user_input
            break
    
    # Reconstruct the image
    reconstruct_image(args.image_path, args.model_path, args.output_dir)

if __name__ == "__main__":
    main()