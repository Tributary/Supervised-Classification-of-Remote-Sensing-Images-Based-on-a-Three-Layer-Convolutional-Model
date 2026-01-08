import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RemoteSensingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        for class_idx, class_name in enumerate(tqdm(self.classes, desc="加载类别")):# 加载所有图像路径和标签
            class_dir = os.path.join(root_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            for img_name in image_files:
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):# 获取图像和标签
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:    # 应用数据增强和预处理
            image = self.transform(image)
        return image, label
    
class RemoteSensingCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(RemoteSensingCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一个卷积块 ,输入：3通道的RGB图片（64×64×3）,输出：32通道的特征图（32×32×32）
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  #用32个3×3×3的滤波器从3个颜色通道中提取32种特征
            nn.ReLU(),
            nn.BatchNorm2d(32),  #批量归一化,加速训练，提高稳定性
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  #最大池化,保留重要特征
            nn.Dropout(0.25),#随机关闭25%的神经元，防止过拟合
            
            # 第二个卷积块,输入：32通道特征图（32×32×32）,输出：64通道更复杂的特征（16×16×64）
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # 第三个卷积块,输入：64通道特征图（16×16×64）,输出：128通道高级特征（8×8×128）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),#将8×8×128=8192个特征压缩到512个决策因子
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),#随机关闭50%的连接，强力防止过拟合
            nn.Linear(512, num_classes)#输出21个类别的概率
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)#展平操作，将三维特征图变成一维向量
        x = self.classifier(x)
        return x

def train_model(model, train_loader, criterion, optimizer, epoch, num_epochs):
    model.train() # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')# 创建进度条
    
    for batch_idx, (data, target) in enumerate(pbar):#每次处理一个批次（32张图片）
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad() # 1. 清空梯度
        output = model(data) # 2. 前向传播
        loss = criterion(output, target) # 3. 计算损失
        loss.backward()  # 4. 反向传播
        optimizer.step()  # 5. 更新参数
        
        running_loss += loss.item()# 累计损失值
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条信息
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def test_model(model, test_loader, criterion):
    model.eval()# 设置为评估模式
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():# 关闭梯度计算
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, all_preds, all_targets

# 主函数
def main():
    transform = transforms.Compose([     # 数据预处理
        transforms.Resize((64, 64)),#把所有图片统一切成64×64大小
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转图片（50%概率），增加数据多样性
        transforms.RandomRotation(10),#随机旋转图片（±10度），让模型学会从不同角度识别
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#标准化处理，让数值范围更稳定
    ])
    dataset = RemoteSensingDataset(root_dir='Images', transform=transform)
    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # 创建数据加载器 - Windows下设置num_workers=0避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)#每次处理32张图片，打乱顺序，单进程加载
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"类别数: {len(dataset.classes)}")
    print(f"类别名称: {dataset.classes}")
    print("=" * 60)
    
    print("初始化模型...")
    model = RemoteSensingCNN(num_classes=len(dataset.classes)).to(device)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    print("=" * 60)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    #lr=0.001：学习率，每次调整的步长，weight_decay=1e-4：权重衰减，防止过拟合
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)#每10个epoch将学习率乘以0.1

    num_epochs = 30
    best_acc = 0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    print(f"开始训练，共 {num_epochs} 个epoch...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        print(f"{'='*40}")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch, num_epochs)
        test_loss, test_acc, all_preds, all_targets = test_model(model, test_loader, criterion)
        scheduler.step()# 学习率调整
        train_losses.append(train_loss)#记录
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} 总结:")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2f}%")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  耗时: {epoch_time:.1f}秒")
        if test_acc > best_acc:# 保存最佳模型
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'class_names': dataset.classes,
                'class_to_idx': dataset.class_to_idx
            }, 'best_remote_sensing_model.pth')
            print(f"  ✓ 保存最佳模型，测试准确率: {test_acc:.2f}%")
    
    print("=" * 60)
    print("训练完成!")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print("\n生成训练曲线图...")# 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Training and Test Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(test_accs, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Training and Test Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ 训练曲线已保存为 'training_curves.png'")

    print("\n最终评估结果:")
    print("-" * 40)
    print("\n分类报告:")# 生成分类报告
    print("-" * 40)
    report = classification_report(all_targets, all_preds, target_names=dataset.classes, output_dict=True)
    print(classification_report(all_targets, all_preds, target_names=dataset.classes))
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(dataset.classes))
    plt.xticks(tick_marks, dataset.classes, rotation=45, ha='right')
    plt.yticks(tick_marks, dataset.classes)
    
    # 在矩阵中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ 混淆矩阵已保存为 'confusion_matrix.png'")
    
    # 保存训练结果到文本文件
    with open('training_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"遥感图像分类训练结果\n")
        f.write(f"="*50 + "\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最佳测试准确率: {best_acc:.2f}%\n")
        f.write(f"总epoch数: {num_epochs}\n")
        f.write(f"\n类别列表:\n")
        for i, cls in enumerate(dataset.classes):
            f.write(f"  {i:2d}. {cls}\n")
        f.write(f"\n训练曲线:\n")
        f.write(f"  Epoch\tTrain Loss\tTest Loss\tTrain Acc\tTest Acc\n")
        for i in range(num_epochs):
            f.write(f"  {i+1:3d}\t{train_losses[i]:.4f}\t{test_losses[i]:.4f}\t{train_accs[i]:.2f}%\t{test_accs[i]:.2f}%\n")
    
    print("✓ 训练结果已保存为 'training_results.txt'")
    print(f"1. 模型文件: best_remote_sensing_model.pth")
    print(f"2. 训练曲线: training_curves.png")
    print(f"3. 混淆矩阵: confusion_matrix.png")
    print(f"4. 训练结果: training_results.txt")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  #多进程启动保护函数
    main()