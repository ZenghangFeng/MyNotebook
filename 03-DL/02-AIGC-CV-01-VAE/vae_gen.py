import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 潜在空间的均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出在[0,1]范围内
        )

    def encode(self, x):
        """编码器：将输入映射到潜在空间的均值和方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧：从分布中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码器：从潜在变量重构数据"""
        return self.decoder(z)

    def forward(self, x):
        # 编码
        mu, logvar = self.encode(x)
        # 重参数化
        z = self.reparameterize(mu, logvar)
        # 解码
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE损失函数：重构损失 + KL散度"""
    # 重构损失（二元交叉熵）
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL散度损失
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss


def load_mnist_data(batch_size=128):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 展平为向量
    ])

    # 训练集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def train_vae(model, train_loader, test_loader, epochs=50, learning_rate=1e-3):
    """训练VAE模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    recon_losses = []
    kl_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0

        # 训练阶段
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # 前向传播
            recon_batch, mu, logvar = model(data)

            # 计算损失
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)

            # 反向传播
            loss.backward()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 每100个batch打印一次训练信息
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_recon_loss = train_recon_loss / len(train_loader.dataset)
        avg_kl_loss = train_kl_loss / len(train_loader.dataset)

        train_losses.append(avg_train_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)

        # 测试阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss, _, _ = vae_loss(recon_batch, data, mu, logvar)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader.dataset)

        print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}, '
              f'Test loss: {avg_test_loss:.4f}, '
              f'Recon loss: {avg_recon_loss:.4f}, KL loss: {avg_kl_loss:.4f}')

    # 绘制损失曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, label='Reconstruction Loss', color='orange')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(kl_losses, label='KL Loss', color='green')
    plt.title('KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, train_losses, recon_losses, kl_losses


def generate_samples(model, num_samples=64, latent_dim=20):
    """从潜在空间采样生成新样本"""
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z).cpu()

    return samples


def interpolate_latent_space(model, z1, z2, num_steps=10):
    """在潜在空间进行插值"""
    model.eval()
    with torch.no_grad():
        # 创建插值点
        interpolations = []
        for alpha in np.linspace(0, 1, num_steps):
            z = alpha * z1 + (1 - alpha) * z2
            sample = model.decode(z)
            interpolations.append(sample.cpu())

    return interpolations


def visualize_results(model, test_loader, latent_dim=20):
    """可视化结果：原始图像、重构图像和生成图像"""
    model.eval()

    # 获取一些测试样本
    data_iter = iter(test_loader)
    test_images, _ = next(data_iter)
    test_images = test_images[:8].to(device)

    with torch.no_grad():
        # 重构图像
        recon_images, mu, logvar = model(test_images)

        # 生成新图像
        generated_images = generate_samples(model, num_samples=8, latent_dim=latent_dim)

    # 可视化
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))

    # 原始图像
    for i in range(8):
        axes[0, i].imshow(test_images[i].cpu().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original Images')

    # 重构图像
    for i in range(8):
        axes[1, i].imshow(recon_images[i].cpu().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed Images')

    # 生成图像
    for i in range(8):
        axes[2, i].imshow(generated_images[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Generated Images')

    plt.tight_layout()
    plt.show()

    # 潜在空间插值可视化
    with torch.no_grad():
        # 选择两个不同的潜在向量
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        # 生成插值
        interpolations = interpolate_latent_space(model, z1, z2, num_steps=10)

    # 显示插值结果
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(interpolations[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Step {i + 1}')

    plt.suptitle('Latent Space Interpolation', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    # 超参数设置
    batch_size = 128
    epochs = 30
    learning_rate = 1e-3
    input_dim = 784  # 28x28
    hidden_dim = 400
    latent_dim = 20  # 潜在空间维度

    print("=" * 60)
    print("VAE手写数字生成实验")
    print("=" * 60)

    # 加载数据
    print("加载MNIST数据集...")
    train_loader, test_loader = load_mnist_data(batch_size=batch_size)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 创建模型
    print("创建VAE模型...")
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("开始训练模型...")
    model, train_losses, recon_losses, kl_losses = train_vae(
        model, train_loader, test_loader, epochs=epochs, learning_rate=learning_rate
    )

    # 可视化结果
    print("可视化结果...")
    visualize_results(model, test_loader, latent_dim=latent_dim)

    # 生成更多样本
    print("生成新的手写数字...")
    samples = generate_samples(model, num_samples=100, latent_dim=latent_dim)

    # 显示生成的图像网格
    plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(samples[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.suptitle('100 Generated Handwritten Digits', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'vae_mnist.pth')
    print("模型已保存到 vae_mnist.pth")

    return model


if __name__ == "__main__":
    model = main()