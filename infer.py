# ============================================
# ResNet U-Net Inference 
# ============================================

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ============================================
# Model Definition
# ============================================
class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, channels, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):
    """Encoder with Residual Blocks"""
    def __init__(self, in_channels, out_channels, num_res_blocks=2, dropout=0.1):
        super(EncoderBlock, self).__init__()
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(out_channels, dropout))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    """Decoder with Simple Skip Connections"""
    def __init__(self, in_channels, skip_channels, out_channels, num_res_blocks=2, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(out_channels, dropout))
        self.res_blocks = nn.Sequential(*res_blocks)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res_blocks(x)
        return x


class ResNetUNet(nn.Module):
    """ResNet-based U-Net"""
    def __init__(self, base_channels=64, dropout=0.1):
        super(ResNetUNet, self).__init__()
        
        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (4 levels)
        self.enc1 = EncoderBlock(base_channels, base_channels, num_res_blocks=2, dropout=dropout)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, num_res_blocks=2, dropout=dropout)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, num_res_blocks=3, dropout=dropout)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8, num_res_blocks=3, dropout=dropout)
        
        # Bottleneck
        bottleneck_blocks = []
        for _ in range(4):
            bottleneck_blocks.append(ResidualBlock(base_channels * 8, dropout))
        self.bottleneck = nn.Sequential(*bottleneck_blocks)
        
        # Decoder (4 levels)
        self.dec4 = DecoderBlock(base_channels * 8, base_channels * 8, base_channels * 4, num_res_blocks=3, dropout=dropout)
        self.dec3 = DecoderBlock(base_channels * 4, base_channels * 4, base_channels * 2, num_res_blocks=2, dropout=dropout)
        self.dec2 = DecoderBlock(base_channels * 2, base_channels * 2, base_channels, num_res_blocks=2, dropout=dropout)
        self.dec1 = DecoderBlock(base_channels, base_channels, base_channels, num_res_blocks=2, dropout=dropout)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 7, 1, 3)
        )
    
    def forward(self, x):
        # Initial
        x0 = self.init_conv(x)
        
        # Encoder
        x1, p1 = self.enc1(x0)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with simple skip connections
        d4 = self.dec4(b, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        
        # Output with residual
        out = self.output(d1)
        out = x + out
        out = torch.clamp(out, 0, 1)
        
        return out


# ============================================
# Inference Functions
# ============================================
def load_model(model_path):
    print(f"Loading model from: {model_path}")
    
    model = ResNetUNet(base_channels=64, dropout=0.1).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 다양한 형태의 checkpoint 지원
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   Loaded ResNet U-Net")
    elif 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
        print("   Loaded ResNet U-Net (from GAN checkpoint)")
    else:
        model.load_state_dict(checkpoint)
        print("   Loaded ResNet U-Net (direct state dict)")
    
    if 'psnr' in checkpoint:
        print(f"   Model PSNR: {checkpoint['psnr']:.2f} dB")
    if 'epoch' in checkpoint:
        print(f"   Trained Epochs: {checkpoint['epoch']}")
    
    model.eval()
    return model


def process_image(model, image_path, output_path, use_fp16=True):
    blur_img = Image.open(image_path).convert('RGB')
    original_size = blur_img.size
    
    transform = transforms.ToTensor()
    blur_tensor = transform(blur_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_fp16:
            with autocast():
                output = model(blur_tensor)
        else:
            output = model(blur_tensor)
    
    output = torch.clamp(output, 0, 1)
    output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    
    if output_img.size != original_size:
        output_img = output_img.resize(original_size, Image.LANCZOS)
    
    output_img.save(output_path)
    
    return blur_img, output_img


def batch_inference(model, test_dir, result_dir, use_fp16=True):
    os.makedirs(result_dir, exist_ok=True)
    
    test_path = Path(test_dir)
    image_files = list(test_path.glob('*.png')) + list(test_path.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"No image files found in {test_dir}")
        return
    
    print(f"\n Found {len(image_files)} images in {test_dir}")
    print(f" Results will be saved to {result_dir}")
    print(f"FP16 Mode: {'Enabled' if use_fp16 else 'Disabled'}")
    print("="*70)
    
    start_time = time.time()
    
    for img_file in tqdm(image_files, desc="Processing"):
        input_path = str(img_file)
        output_filename = f"deblurred_{img_file.name}"
        output_path = os.path.join(result_dir, output_filename)
        
        try:
            process_image(model, input_path, output_path, use_fp16)
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    total_time = time.time() - start_time
    
    print("="*70)
    print(f"Processing complete!")
    print(f"   Total images: {len(image_files)}")
    print(f"   Time: {total_time:.2f}s ({total_time/len(image_files):.2f}s per image)")
    print(f"   Output: {result_dir}")
    print("="*70)


def main():
    MODEL_PATH = r"c:\Users\KUEEE03\Desktop\mw\model\best_resnet_unet.pth"
    TEST_DIR = r"c:\Users\KUEEE03\Desktop\mw\test_data"
    RESULT_DIR = r"c:\Users\KUEEE03\Desktop\mw\result"
    USE_FP16 = True
    
    print("\n" + "="*70)
    print("RESNET U-NET INFERENCE ")
    print("="*70)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Input: {TEST_DIR}")
    print(f"  Output: {RESULT_DIR}")
    print(f"  FP16: {USE_FP16}")
    print("="*70)
    
    model = load_model(MODEL_PATH)
    batch_inference(model, TEST_DIR, RESULT_DIR, USE_FP16)


if __name__ == "__main__":
    main()