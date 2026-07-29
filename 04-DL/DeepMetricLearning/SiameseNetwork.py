import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# ==========================================================================================
# 创建配对数据集
# ==========================================================================================
class SiameseMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, num_pairs=50000, train=True):
        self.mnist_dataset = mnist_dataset
        self.num_pairs = num_pairs
        self.train = train

        # 根据数字索引将数据分组
        self.digit_indices = [[] for _ in range(10)]
        for idx, (img, label) in enumerate(mnist_dataset):
            self.digit_indices[label].append(idx)

        self.pairs = []
        self.labels = []
        self._generate_pairs()

    def _generate_pairs(self):
        for _ in range(self.num_pairs):
            digit1 = np.random.randint(0, 10)
            digit2 = np.random.randint(0, 10)

            idx1 = np.random.choice(self.digit_indices[digit1])
            idx2 = np.random.choice(self.digit_indices[digit2])

            img1, _ = self.mnist_dataset[idx1]
            img2, _ = self.mnist_dataset[idx2]

            # 标签：1 表示不同，0 表示相同
            label = 0 if digit1 == digit2 else 1
            self.pairs.append((img1, img2))
            self.labels.append(label)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ==========================================================================================
# 定义孪生网络结构
# ==========================================================================================
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64 * 7 * 7, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_net = BaseNetwork()

    def forward(self, x1, x2):
        h1 = self.base_net(x1)
        h2 = self.base_net(x2)
        # 计算欧氏距离（L2 范数）
        distance = F.pairwise_distance(h1, h2, p=2)
        return distance


# ==========================================================================================
# 对比损失函数
# ==========================================================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label=1 表示不相似，label=0 表示相似
        loss = (1 - label) * torch.pow(distance, 2) / 2 + \
               label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) / 2
        return loss.mean()


# ==========================================================================================
# 训练准备
# ==========================================================================================
# 数据预处理：转换为 Tensor 并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 训练集和测试集
train_mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_mnist = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 创建配对数据集
train_pair_dataset = SiameseMNISTDataset(train_mnist, num_pairs=20000, train=True)
test_pair_dataset = SiameseMNISTDataset(test_mnist, num_pairs=2000, train=False)

train_loader = DataLoader(train_pair_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_pair_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ==========================================================================================
# 训练循环
# ==========================================================================================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for img1, img2, label in progress:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        distance = model(img1, img2)
        loss = criterion(distance, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)


def evaluate():
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            distance = model(img1, img2)
            # 判断相似：距离小于阈值（这里取 1.0）则预测为相似(0)，否则不相似(1)
            pred = (distance > 1.0).float()
            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)
    accuracy = total_correct / total_samples
    return accuracy


# 训练 5 个 epoch
epochs = 5
for epoch in range(epochs):
    train_loss = train_one_epoch(epoch)
    acc = evaluate()
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Accuracy: {acc:.4f}")


# ==========================================================================================
#  测试示例：随机挑选两张图片并预测相似性
# ==========================================================================================
# 从测试集中随机取一对
def visualize_pair(idx):
    img1, img2, label = test_pair_dataset[idx]
    model.eval()
    with torch.no_grad():
        d = model(img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device))
        pred = "similar" if d.item() < 1.0 else "different"

    # 显示图片
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img1.squeeze(), cmap='gray')
    axes[0].set_title('Image 1')
    axes[1].imshow(img2.squeeze(), cmap='gray')
    axes[1].set_title('Image 2')
    plt.suptitle(f"Ground Truth: {'similar' if label == 0 else 'different'}, Pred: {pred}, Distance: {d.item():.4f}")
    plt.show()


# 举例：测试几个样本对
for i in [10, 50, 100, 200]:
    visualize_pair(i)