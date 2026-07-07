import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==================== 数据准备 (优化版本) ====================
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

# 定义图像预处理变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 统一调整为128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义分割掩码的预处理变换
# 注意：需要先将掩码转换为灰度图('L')，然后转换为Tensor，最后将像素值0、1、2转换为0、1
target_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.PILToTensor(),  # 直接转换为Tensor，保持数值
    lambda x: (x.squeeze().float() > 0).float().unsqueeze(0)  # 将掩码转换为二值(宠物/背景)
])

# 直接使用torchvision的OxfordIIITPet加载数据集[citation:5]
# 关键参数说明[citation:1]：
# - target_types='segmentation': 加载分割掩码
# - download=True: 如果数据集不存在则自动下载
train_dataset = OxfordIIITPet(
    root='./data',
    split='trainval',
    target_types='segmentation',
    download=True,
    transform=transform,
    target_transform=target_transform
)

test_dataset = OxfordIIITPet(
    root='./data',
    split='test',
    target_types='segmentation',
    download=False,  # 测试集默认不重复下载
    transform=transform,
    target_transform=target_transform
)

# 创建DataLoader
# 优化建议：根据CPU核心数设置num_workers，开启pin_memory加速GPU传输[citation:6]
import os
num_workers = min(0, os.cpu_count())  # 根据CPU核心数动态设置

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


# ==================== U-Net 模型定义 ====================
class DoubleConv(nn.Module):
    """双卷积块：卷积 -> BN -> ReLU -> 卷积 -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器（下采样路径）
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器（上采样路径）
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 编码器
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器（反转跳跃连接）
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # 上采样
            x = self.ups[idx](x)

            # 跳跃连接
            skip_connection = skip_connections[idx // 2]

            # 调整尺寸（处理尺寸不匹配）
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # 拼接特征
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        # 最终输出
        return self.final_conv(x)


# ==================== 训练配置 ====================
def dice_coefficient(pred, target, smooth=1e-6):
    """计算Dice系数（分割评估指标）"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class DiceBCELoss(nn.Module):
    """Dice损失 + BCE损失的组合"""

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # BCE损失
        bce = F.binary_cross_entropy_with_logits(inputs, targets)

        # Dice损失
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

        return bce + dice


# 初始化模型、损失函数和优化器
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    dice_scores = []

    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算指标
        running_loss += loss.item()

        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            dice = dice_coefficient(preds.float(), masks)
            dice_scores.append(dice.item())

        pbar.set_postfix({'Loss': loss.item(), 'Dice': dice.item()})

    return running_loss / len(dataloader), np.mean(dice_scores)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            dice = dice_coefficient(preds.float(), masks)
            dice_scores.append(dice.item())

            pbar.set_postfix({'Loss': loss.item(), 'Dice': dice.item()})

    return running_loss / len(dataloader), np.mean(dice_scores)


# ==================== 训练循环 ====================
num_epochs = 1
train_losses, val_losses = [], []
train_dices, val_dices = [], []

print("开始训练U-Net模型...")
print("=" * 50)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # 训练
    train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_dices.append(train_dice)

    # 验证
    val_loss, val_dice = validate(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    val_dices.append(val_dice)

    # 学习率调度
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
    print("-" * 50)


# ==================== 可视化结果 ====================
def visualize_results(model, dataloader, device, num_samples=3):
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            if idx >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            # 反归一化图像
            img_np = images[0].cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            # 真实掩码
            true_mask = masks[0, 0].cpu().numpy()

            # 预测掩码
            pred_mask = preds[0, 0].cpu().numpy()

            # 绘制
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title('Input Image')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(true_mask, cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(pred_mask, cmap='gray')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')

    plt.suptitle('U-Net Segmentation Results', fontsize=16)
    plt.tight_layout()
    plt.show()


# 可视化训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_dices, label='Train Dice')
plt.plot(val_dices, label='Val Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficient During Training')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 显示分割结果
print("\n显示分割结果示例:")
visualize_results(model, test_loader, device, num_samples=3)


# ==================== 模型评估 ====================
def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            # 计算Dice系数
            dice = dice_coefficient(preds.float(), masks)
            total_dice += dice.item()

            # 计算IoU
            pred_flat = preds.view(-1)
            mask_flat = masks.view(-1)
            intersection = (pred_flat & mask_flat).float().sum()
            union = (pred_flat | mask_flat).float().sum()
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.item()

    return total_dice / num_batches, total_iou / num_batches


# 最终评估
final_dice, final_iou = evaluate_model(model, test_loader, device)
print(f"\n最终评估结果:")
print(f"Dice Coefficient: {final_dice:.4f}")
print(f"IoU Score: {final_iou:.4f}")

# ==================== 保存模型 ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_dices': train_dices,
    'val_dices': val_dices
}, 'unet_pet_segmentation.pth')

print("模型已保存为 'unet_pet_segmentation.pth'")


# ==================== 模型应用示例 ====================
def predict_single_image(model, image_path, device):
    """对单张图像进行分割预测"""
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output) > 0.5

    # 转换为numpy
    input_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
    input_np = input_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    input_np = np.clip(input_np, 0, 1)

    pred_np = pred[0, 0].cpu().numpy()

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(input_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(pred_np, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    # 叠加显示
    axes[2].imshow(input_np)
    axes[2].imshow(pred_np, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# 如果本地有宠物图片，可以使用以下代码进行预测
# predict_single_image(model, 'your_pet_image.jpg', device)