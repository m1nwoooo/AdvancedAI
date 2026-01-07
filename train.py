# ============================================
#  ResNet U-Net 
# CNN + Residual 
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import random
import time
from pathlib import Path
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# 1. Dataset
# ============================================
class GoPro_Dataset(Dataset):
    def __init__(self, root_dir, crop_size=256, is_train=True):
        self.blur_paths = []
        self.sharp_paths = []
        self.crop_size = crop_size
        self.is_train = is_train
        
        root_path = Path(root_dir)
        scene_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('GOPR')])
        print(f"Found {len(scene_dirs)} scenes")
        
        for scene_dir in scene_dirs:
            blur_dir = scene_dir / 'blur'
            sharp_dir = scene_dir / 'sharp'
            if not blur_dir.exists() or not sharp_dir.exists():
                continue
            blur_files = sorted(blur_dir.glob('*.png'))
            for blur_file in blur_files:
                sharp_file = sharp_dir / blur_file.name
                if sharp_file.exists():
                    self.blur_paths.append(str(blur_file))
                    self.sharp_paths.append(str(sharp_file))
        
        print(f"Loaded {len(self.blur_paths)} pairs")
    
    def __len__(self):
        return len(self.blur_paths)
    
    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_paths[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_paths[idx]).convert('RGB')
        
        if self.is_train:
            blur_img, sharp_img = self._augment(blur_img, sharp_img)
        else:
            blur_img = transforms.CenterCrop(self.crop_size)(blur_img)
            sharp_img = transforms.CenterCrop(self.crop_size)(sharp_img)
        
        blur_tensor = transforms.ToTensor()(blur_img)
        sharp_tensor = transforms.ToTensor()(sharp_img)
        
        return blur_tensor, sharp_tensor
    
    def _augment(self, blur, sharp):
        w, h = blur.size
        
        if w >= self.crop_size and h >= self.crop_size:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            blur = blur.crop((x, y, x + self.crop_size, y + self.crop_size))
            sharp = sharp.crop((x, y, x + self.crop_size, y + self.crop_size))
        else:
            blur = transforms.Resize((self.crop_size, self.crop_size))(blur)
            sharp = transforms.Resize((self.crop_size, self.crop_size))(sharp)
        
        if random.random() > 0.5:
            blur = transforms.functional.hflip(blur)
            sharp = transforms.functional.hflip(sharp)
        
        if random.random() > 0.5:
            blur = transforms.functional.vflip(blur)
            sharp = transforms.functional.vflip(sharp)
        
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            blur = transforms.functional.rotate(blur, angle)
            sharp = transforms.functional.rotate(sharp, angle)
        
        return blur, sharp


# ============================================
# 2. CNN Architecture 
# ============================================
class ResidualBlock(nn.Module):
    """Residual Block """
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
        out += residual  # Skip connection 
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
        
        self.pool = nn.MaxPool2d(2, 2)  # Max Pooling
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    """Decoder with Simple Skip Connections"""
    def __init__(self, in_channels, skip_channels, out_channels, num_res_blocks=2, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        
        # Skip connection: 단순 concatenation
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
        x = torch.cat([x, skip], dim=1)  # 단순 concatenation
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
        out = x + out  # Residual learning
        out = torch.clamp(out, 0, 1)
        
        return out


# ============================================
# 3. Loss Functions 
# ============================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))
        self.slice2 = nn.Sequential(*list(vgg[4:9]))
        self.slice3 = nn.Sequential(*list(vgg[9:16]))
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, output, target):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(output.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(output.device)
        
        output = (output - mean) / std
        target = (target - mean) / std
        
        out_f1 = self.slice1(output)
        out_f2 = self.slice2(out_f1)
        out_f3 = self.slice3(out_f2)
        
        tar_f1 = self.slice1(target)
        tar_f2 = self.slice2(tar_f1)
        tar_f3 = self.slice3(tar_f2)
        
        loss = (
            nn.functional.l1_loss(out_f1, tar_f1) +
            nn.functional.l1_loss(out_f2, tar_f2) +
            nn.functional.l1_loss(out_f3, tar_f3)
        )
        
        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, x):
        h_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        w_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv_loss = torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))
        return tv_loss


class CombinedLoss(nn.Module):
    """Combined loss"""
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1, lambda_tv=0.0001):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.tv_loss = TotalVariationLoss()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_tv = lambda_tv
    
    def forward(self, output, target):
        l1 = self.l1_loss(output, target)
        perceptual = self.perceptual_loss(output, target)
        tv = self.tv_loss(output)
        
        total = (
            self.lambda_l1 * l1 +
            self.lambda_perceptual * perceptual +
            self.lambda_tv * tv
        )
        
        return total, l1, perceptual, tv


# ============================================
# 4. Training
# ============================================
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    
    epoch_loss = 0.0
    epoch_l1 = 0.0
    epoch_perceptual = 0.0
    epoch_tv = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (blur, sharp) in enumerate(pbar):
        blur = blur.to(device, non_blocking=True)
        sharp = sharp.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            output = model(blur)
            total_loss, l1_loss, perceptual_loss, tv_loss = criterion(output, sharp)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += total_loss.item()
        epoch_l1 += l1_loss.item()
        epoch_perceptual += perceptual_loss.item()
        epoch_tv += tv_loss.item()
        
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'Perc': f'{perceptual_loss.item():.4f}'
        })
    
    n = len(dataloader)
    return epoch_loss/n, epoch_l1/n, epoch_perceptual/n, epoch_tv/n


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0.0
    
    with torch.no_grad():
        for blur, sharp in tqdm(dataloader, desc="Validation", leave=False):
            blur = blur.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)
            
            with autocast():
                output = model(blur)
            
            output = torch.clamp(output, 0, 1)
            sharp = torch.clamp(sharp, 0, 1)
            
            psnr = calculate_psnr(output, sharp)
            total_psnr += psnr
    
    return total_psnr / len(dataloader)


# ============================================
# 5. Main Training
# ============================================
def train_model():
    BATCH_SIZE = 8
    NUM_EPOCHS = 350
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.1
    CROP_SIZE = 256
    NUM_WORKERS = 4
    PATIENCE = 15
    BASE_CHANNELS = 64
    
    DATASET_ROOT = r"D:\dataset"
    TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
    TEST_DIR = os.path.join(DATASET_ROOT, "test")
    SAVE_DIR = r"c:\Users\KUEEE03\Desktop\mw\model"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("  RESNET U-NET ")
    print("="*70)
    print(f"  FP16 Mixed Precision: Enabled ⚡")
    print(f"  Regularization:")
    print(f"    - Weight Decay: {WEIGHT_DECAY}")
    print(f"    - Dropout: {DROPOUT}")
    print(f"    - Batch Normalization")
    print(f"  Loss Functions:")
    print(f"    - L1 Loss")
    print(f"    - Perceptual Loss (VGG)")
    print(f"    - Total Variation Loss")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Base Channels: {BASE_CHANNELS}")
    print("="*70)
    
    # Dataset
    train_dataset = GoPro_Dataset(TRAIN_DIR, crop_size=CROP_SIZE, is_train=True)
    test_dataset = GoPro_Dataset(TEST_DIR, crop_size=CROP_SIZE, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Train: {len(train_loader)} batches ({len(train_dataset)} images)")
    print(f"Test: {len(test_loader)} batches ({len(test_dataset)} images)")
    
    # Model
    model = ResNetUNet(base_channels=BASE_CHANNELS, dropout=DROPOUT).to(device)
    
    model_params = sum(p.numel() for p in model.parameters())
    print(f"   Model Parameters: {model_params:,}")
    
    # Loss
    criterion = CombinedLoss(
        lambda_l1=1.0,
        lambda_perceptual=0.2,
        lambda_tv=0.00005
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), 
                          lr=LEARNING_RATE, 
                          betas=(0.9, 0.999),
                          weight_decay=WEIGHT_DECAY)
    
    scaler = GradScaler()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=True
    )
    
    best_psnr = 0.0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        total_loss, l1_loss, perc_loss, tv_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  Total Loss: {total_loss:.4f}")
        print(f"    - L1: {l1_loss:.4f}")
        print(f"    - Perceptual: {perc_loss:.4f}")
        print(f"    - Total Variation: {tv_loss:.6f}")
        print(f"  Time: {epoch_time:.1f}s ⚡")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch % 5 == 0:
            val_psnr = validate(model, test_loader, device)
            print(f"  Validation PSNR: {val_psnr:.2f} dB")
            
            scheduler.step(val_psnr)
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                patience_counter = 0
                
                model_path = os.path.join(SAVE_DIR, 'best_resnet_unet_no_gan.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': best_psnr,
                }, model_path)
                print(f"  Best model saved! PSNR: {best_psnr:.2f} dB")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{PATIENCE})")
                
                if patience_counter >= PATIENCE:
                    print(f"\n Early stopping at epoch {epoch}")
                    break
            
            checkpoint_path = os.path.join(SAVE_DIR, f'checkpoint_resnet_unet_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': best_psnr,
            }, checkpoint_path)
            print(f"  Checkpoint saved: epoch_{epoch}.pth")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Models saved to: {SAVE_DIR}")
    print("="*70)
    
    return model, best_psnr


if __name__ == "__main__":
    trained_model, best_psnr = train_model()