import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os

class RemoteSensingCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(RemoteSensingCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # 第三个卷积块
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
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SlidingWindowPredictor:
    def __init__(self, model_path='best_remote_sensing_model.pth'):
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件: {model_path}")
        
        # 修复安全警告：先尝试安全模式，如果失败则使用传统模式
        try:
            # 尝试使用安全模式加载
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            print("✅ 使用安全模式加载模型")
        except Exception as e:
            # 如果安全模式失败，使用传统模式
            print("⚠️ 安全模式加载失败，使用传统模式")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        
        print(f"📊 加载模型信息:")
        print(f"   - 类别数量: {self.num_classes}")
        print(f"   - 最佳准确率: {checkpoint.get('best_acc', 'N/A')}%")
        print(f"   - 可识别类别: {', '.join(self.class_names)}")
        
        # 创建模型实例并加载权重
        self.model = RemoteSensingCNN(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        # 预处理（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def sliding_window(self, image, window_size=256, stride=128):
        """滑动窗口生成器"""
        h, w = image.shape[:2]
        
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                yield (x, y, image[y:y+window_size, x:x+window_size])
    
    def predict_single_window(self, window_image):
        """对单个窗口进行预测"""
        # 转换窗口为PIL图像
        window_pil = Image.fromarray(window_image)
        
        # 预处理
        input_tensor = self.transform(window_pil).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = self.class_names[predicted_idx.item()]
            
            return predicted_class, confidence
    
    def predict_big_image(self, big_image_path, window_size=256, stride=128, confidence_threshold=0.5):
        if not os.path.exists(big_image_path):
            raise FileNotFoundError(f"❌ 找不到图片文件: {big_image_path}")
        
        # 读取大图片
        big_image = cv2.imread(big_image_path)
        if big_image is None:
            raise ValueError(f"❌ 无法读取图片文件: {big_image_path}")
        
        big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2RGB)
        h, w = big_image.shape[:2]
        
        print(f"📐 图片尺寸: {w} × {h}")
        print(f"🪟 窗口大小: {window_size} × {window_size}")
        print(f"🚶 滑动步长: {stride}")
        print(f"🎯 置信度阈值: {confidence_threshold}")
        
        # 如果图片太小，调整窗口大小
        if h < window_size or w < window_size:
            print("⚠️ 图片尺寸较小，自动调整窗口大小")
            window_size = min(h, w) // 2
            stride = window_size // 2
        
        # 存储识别结果
        results = []
        window_count = 0
        
        print("⏳ 正在分析图片区域...")
        # 滑动窗口处理
        for x, y, window in self.sliding_window(big_image, window_size, stride):
            window_count += 1
            
            # 对当前窗口进行预测
            predicted_class, confidence = self.predict_single_window(window)
            
            # 只保留高置信度的结果
            if confidence > confidence_threshold:
                results.append({
                    'x': x, 'y': y, 
                    'class': predicted_class,
                    'confidence': confidence,
                    'window_size': window_size
                })
            
            # 显示进度
            if window_count % 10 == 0:
                print(f"  已处理 {window_count} 个窗口，找到 {len(results)} 个高置信度区域")
        
        print(f"✅ 处理完成！共分析 {window_count} 个窗口，找到 {len(results)} 个高置信度区域")
        return results, big_image, window_count
    
    def visualize_results(self, big_image, results, window_size=256, output_path='result/result.png'):
        """可视化识别结果"""
        if len(results) == 0:
            print("❌ 未找到任何高置信度的区域")
            print("💡 建议: 降低置信度阈值或检查图片内容")
            return None
        
        plt.figure(figsize=(15, 10))
        
        # 创建颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        color_map = {cls: colors[i] for i, cls in enumerate(self.class_names)}
        
        # 显示原图
        plt.imshow(big_image)
        plt.title('大图片区域识别结果', fontsize=16, fontweight='bold')
        
        # 绘制识别框
        for result in results:
            x, y = result['x'], result['y']
            class_name = result['class']
            confidence = result['confidence']
            
            # 绘制矩形框
            rect = Rectangle((x, y), window_size, window_size, 
                           linewidth=2, edgecolor=color_map[class_name], 
                           facecolor='none', alpha=0.8)
            plt.gca().add_patch(rect)
            
            # 添加标签
            plt.text(x + 5, y + 15, f'{class_name}\n({confidence:.2f})', 
                    fontsize=8, color=color_map[class_name], fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        plt.axis('off')
        plt.tight_layout()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存结果图片
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图像以避免内存泄漏
        print(f"📸 结果图已保存为: {output_path}")
        
        # 生成统计报告
        self.generate_report(results, output_path.replace('.png', '_report.txt'))
        
        return output_path
    
    def generate_report(self, results, report_path):
        """生成详细统计报告"""
        from collections import Counter
        
        # 统计各类别数量
        class_counter = Counter([r['class'] for r in results])
        total_regions = len(results)
        
        print("\n📈 区域识别统计报告:")
        print("=" * 50)
        for cls, count in class_counter.most_common():
            percentage = (count / total_regions) * 100
            print(f"  {cls:<20}: {count:>3} 个区域 ({percentage:>5.1f}%)")
        
        print(f"  总计: {total_regions} 个识别区域")
        print("=" * 50)
        
        # 保存报告到文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("区域识别统计报告\n")
            f.write("=" * 50 + "\n")
            for cls, count in class_counter.most_common():
                percentage = (count / total_regions) * 100
                f.write(f"{cls:<20}: {count:>3} 个区域 ({percentage:>5.1f}%)\n")
            f.write(f"总计: {total_regions} 个识别区域\n")
        
        print(f"📄 详细报告已保存为: {report_path}")

def batch_predict_images():
    WINDOW_SIZE =256       # 窗口大小
    STRIDE = 128           # 滑动步长  
    CONFIDENCE_THRESHOLD = 0.7
    model_path = 'best_remote_sensing_model.pth'
    test_dir = 'test'
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    predictor = SlidingWindowPredictor(model_path)
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]
    image_files.sort()  # 按文件名排序
    
    for i, image_file in enumerate(image_files, 1):# 批处理
        image_path = os.path.join(test_dir, image_file)
        image_name = os.path.splitext(image_file)[0]  # 获取文件名（不含扩展名）
        
        print(f"\n[{i}/{len(image_files)}] 处理图片: {image_file}")
        
        try:
            # 进行预测
            results, big_image, total_windows = predictor.predict_big_image(
                image_path, 
                window_size=WINDOW_SIZE, 
                stride=STRIDE, 
                confidence_threshold=CONFIDENCE_THRESHOLD
            )
            
            # 保存结果
            output_path = os.path.join(result_dir, f"{image_name}_result.png")
            predictor.visualize_results(big_image, results, WINDOW_SIZE, output_path)
            
            print(f"✅ 图片 {image_file} 处理完成")
            
        except Exception as e:
            print(f"❌ 处理图片 {image_file} 时出错: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("🎉 批量处理完成！")
    print(f"📁 所有结果已保存到 {result_dir} 文件夹")

if __name__ == '__main__':
    batch_predict_images()