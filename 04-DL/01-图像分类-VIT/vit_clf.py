import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ========== 1. 数据准备模块 ==========
class CatDogDataset(Dataset):
    """猫狗数据集类"""

    def __init__(self, data_dir, transform=None, train=True):
        """
        Args:
            data_dir: 数据目录路径
            transform: 数据增强转换
            train: 是否为训练模式
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []

        # 猫的类别标签为0，狗的类别标签为1
        cat_dir = os.path.join(data_dir, 'cats')
        dog_dir = os.path.join(data_dir, 'dogs')

        # 加载猫的图像
        if os.path.exists(cat_dir):
            for img_name in os.listdir(cat_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(cat_dir, img_name))
                    self.labels.append(0)  # 猫: 0

        # 加载狗的图像
        if os.path.exists(dog_dir):
            for img_name in os.listdir(dog_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(dog_dir, img_name))
                    self.labels.append(1)  # 狗: 1

        print(f"数据集加载完成: {len(self.image_paths)} 张图像")
        print(f"猫: {self.labels.count(0)} 张, 狗: {self.labels.count(1)} 张")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 应用数据增强
        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loaders(data_dir, batch_size=32, img_size=224):
    """创建训练、验证和测试数据加载器"""

    # 训练数据增强（较强）
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试数据增强（较弱）
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建完整数据集
    full_dataset = CatDogDataset(data_dir, transform=train_transform, train=True)

    # 划分数据集（80%训练，10%验证，10%测试）
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 将验证集的数据增强改为val_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")

    return train_loader, val_loader, test_loader


# ========== 2. 简化ViT模型 ==========
class PatchEmbedding(nn.Module):
    """图像分块嵌入"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_patches, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, embed_dim=384, num_heads=6, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CatDogViT(nn.Module):
    """专门用于猫狗分类的ViT模型"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0,
                 dropout=0.1, attention_dropout=0.1):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        # 1. 图像分块嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # 2. 位置编码
        self.pos_encoding = PositionalEncoding(self.num_patches, embed_dim, dropout)

        # 3. Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 4. 归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 5. 分类头 - 二分类
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # 输出2个类别
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # 图像分块嵌入
        x = self.patch_embed(x)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer编码器
        for block in self.blocks:
            x = block(x)

        # 归一化
        x = self.norm(x)

        # 取[CLS]令牌用于分类
        cls_token = x[:, 0]

        # 分类头
        logits = self.head(cls_token)

        return logits

    def predict_proba(self, x):
        """返回概率预测"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def predict(self, x):
        """返回类别预测"""
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


# ========== 3. 训练和评估模块 ==========
class ViTTrainer:
    """ViT模型训练器"""

    def __init__(self, model, device, lr=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device

        # 损失函数（二分类交叉熵）
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-6
        )

        # 跟踪训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch + 1}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            total_loss += loss.item() * images.size(0)
            total_correct += correct
            total_samples += images.size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / images.size(0)
            })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(avg_acc)

        return avg_loss, avg_acc

    def validate(self, val_loader, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'验证 Epoch {epoch + 1}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                total_loss += loss.item() * images.size(0)
                total_correct += correct
                total_samples += images.size(0)

                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': correct / images.size(0)
                })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)

        return avg_loss, avg_acc

    def train(self, train_loader, val_loader, num_epochs=20):
        """完整训练过程"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_acc = self.validate(val_loader, epoch)

            # 学习率调度
            self.scheduler.step()

            print(f"\n训练结果: 损失={train_loss:.4f}, 准确率={train_acc:.4f}")
            print(f"验证结果: 损失={val_loss:.4f}, 准确率={val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, 'best_catdog_vit.pth')
                print(f"保存最佳模型，验证准确率: {val_acc:.4f}")

        print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")

        # 加载最佳模型
        self.load_best_model()

        return self.history

    def load_best_model(self):
        """加载最佳模型"""
        if os.path.exists('best_catdog_vit.pth'):
            checkpoint = torch.load('best_catdog_vit.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("已加载最佳模型")

    def evaluate(self, test_loader):
        """在测试集上评估模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = total_correct / total_samples

        print(f"\n测试集评估结果:")
        print(f"总样本数: {total_samples}")
        print(f"正确分类数: {total_correct}")
        print(f"测试准确率: {accuracy:.4f}")

        return accuracy, all_predictions, all_labels

    def predict_single_image(self, image_path, img_size=224):
        """预测单张图像"""
        # 加载和预处理图像
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model.predict_proba(image_tensor)
            prediction = self.model.predict(image_tensor)

        # 获取类别和概率
        class_idx = prediction.item()
        class_name = "猫" if class_idx == 0 else "狗"
        confidence = probabilities[0, class_idx].item()

        return class_name, confidence, image_tensor


# ========== 4. 可视化模块 ==========
def plot_training_history(history):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率')
    axes[1].plot(history['val_acc'], label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('训练和验证准确率')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=8):
    """可视化预测结果"""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

    # 反归一化图像以便显示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_display = images * std + mean
    images_display = torch.clamp(images_display, 0, 1)

    # 绘制图像和预测结果
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images_display[i].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        true_label = "猫" if labels[i].item() == 0 else "狗"
        pred_label = "猫" if predictions[i].item() == 0 else "狗"
        confidence = probabilities[i, predictions[i]].item()

        color = 'green' if true_label == pred_label else 'red'

        axes[i].set_title(f"真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.2f}",
                          color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150)
    plt.show()


# ========== 5. 主程序 ==========
def main():
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 检查是否有可用的猫狗数据集
    data_dir = "./cat_dog_dataset"  # 修改为你的数据集路径
    if not os.path.exists(data_dir):
        print(f"警告: 数据集目录 {data_dir} 不存在")
        print("请确保数据集结构为:")
        print(f"{data_dir}/cats/ [猫的图像]")
        print(f"{data_dir}/dogs/ [狗的图像]")

        # 如果没有数据集，可以使用torchvision下载示例数据
        print("\n将使用torchvision下载示例数据集...")
        return download_example_dataset(device)

    # 创建数据加载器
    print("加载数据集...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, batch_size=32, img_size=224
    )

    # 创建模型
    print("\n创建ViT模型...")
    model = CatDogViT(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6,
        dropout=0.1,
        attention_dropout=0.1
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建训练器
    trainer = ViTTrainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=1e-4
    )

    # 训练模型
    history = trainer.train(train_loader, val_loader, num_epochs=20)

    # 绘制训练历史
    plot_training_history(history)

    # 在测试集上评估
    test_acc, predictions, labels = trainer.evaluate(test_loader)

    # 可视化预测结果
    visualize_predictions(trainer.model, test_loader, device)

    # 示例：预测单张图像
    print("\n示例预测:")
    # 这里假设有一些测试图像
    test_images = []
    cat_dir = os.path.join(data_dir, 'cats')
    dog_dir = os.path.join(data_dir, 'dogs')

    if os.path.exists(cat_dir):
        cat_images = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_images.extend(cat_images[:2])

    if os.path.exists(dog_dir):
        dog_images = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_images.extend(dog_images[:2])

    if test_images:
        print("\n单张图像预测示例:")
        for img_path in test_images[:4]:
            if os.path.exists(img_path):
                class_name, confidence, _ = trainer.predict_single_image(img_path)
                print(f"图像: {os.path.basename(img_path)}")
                print(f"  预测类别: {class_name}")
                print(f"  置信度: {confidence:.4f}")
                print()

    return trainer


def download_example_dataset(device):
    """下载示例数据集"""
    print("\n下载torchvision的示例数据集...")

    # 创建模拟数据集（实际使用时请替换为真实数据集）
    print("请准备真实猫狗数据集或使用Kaggle数据集")
    print("Kaggle猫狗数据集链接: https://www.kaggle.com/c/dogs-vs-cats")

    # 创建一个简单的演示模型
    model = CatDogViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=4
    )

    # 示例输入
    print("\n创建演示模型...")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 示例前向传播
    demo_input = torch.randn(2, 3, 224, 224).to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(demo_input)
        prob = F.softmax(output, dim=1)

    print(f"示例输入形状: {demo_input.shape}")
    print(f"示例输出形状: {output.shape}")
    print(f"示例预测概率: {prob}")

    return None


# ========== 6. 模型部署和推理示例 ==========
class CatDogClassifier:
    """猫狗分类器封装类，便于部署"""

    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)

        # 加载模型
        self.model = CatDogViT(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=6,
            num_heads=6
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载模型: {model_path}")
        else:
            print("使用随机初始化模型")

        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """预测图像类别"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        class_names = ['猫', '狗']
        confidence = probabilities[0, prediction].item()

        result = {
            'prediction': class_names[prediction],
            'confidence': confidence,
            'probabilities': {
                '猫': probabilities[0, 0].item(),
                '狗': probabilities[0, 1].item()
            }
        }

        return result

    def predict_batch(self, image_paths):
        """批量预测"""
        results = []
        for path in image_paths:
            if os.path.exists(path):
                result = self.predict(path)
                result['image_path'] = path
                results.append(result)

        return results


# ========== 7. 训练脚本入口 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("猫狗分类 Vision Transformer 训练系统")
    print("=" * 60)

    # 运行主程序
    trainer = main()

    print("\n训练完成!")
    print("使用说明:")
    print("1. 最佳模型已保存为 'best_catdog_vit.pth'")
    print("2. 训练历史曲线已保存为 'training_history.png'")
    print("3. 预测可视化已保存为 'predictions_visualization.png'")
    print("\n使用示例:")
    print("```python")
    print("# 加载训练好的模型进行预测")
    print("classifier = CatDogClassifier('best_catdog_vit.pth', 'cuda' if torch.cuda.is_available() else 'cpu')")
    print("result = classifier.predict('your_image.jpg')")
    print("print(f\"预测结果: {result['prediction']}, 置信度: {result['confidence']:.4f}\")")
    print("```")