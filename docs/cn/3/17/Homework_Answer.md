# 课后拓展参考答案

## 练习1：CNN 架构设计与对比实验

**任务描述**：

+ 设计并实现 3 种不同的 CNN 架构，对比它们在同一数据集上的性能表现
+ 实现 LeNet、自定义轻量级网络和稍复杂的网络，分析各自的优缺点
+ 通过实验验证网络深度、卷积核大小、池化策略对模型性能的影响
+ 可视化不同层学到的特征图，理解 CNN 的特征学习过程

**提示**：

+ 使用 CIFAR-10 数据集进行实验，该数据集包含 10 个类别的 32×32 彩色图像
+ 实现详细的训练监控，记录损失、准确率、训练时间等指标
+ 使用 matplotlib 可视化中间层特征图，观察不同层学到的特征
+ 对比不同架构的参数数量、计算复杂度和推理速度

<details>
<summary>点击查看答案</summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

class CNNArchitectureComparator:
    def __init__(self, device=None):
        """CNN架构对比器"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载CIFAR-10数据集
        self.train_loader, self.test_loader = self.load_cifar10()
        
        # 存储实验结果
        self.experiment_results = {}
    
    def load_cifar10(self, batch_size=128):
        """加载CIFAR-10数据集"""
        print("加载CIFAR-10数据集...")
        
        # 数据预处理
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 加载数据集
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                               download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                                 shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                              download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                shuffle=False, num_workers=2)
        
        print(f"训练集大小: {len(trainset)}")
        print(f"测试集大小: {len(testset)}")
        
        return trainloader, testloader
    
    def create_lenet_model(self):
        """创建LeNet风格模型"""
        class LeNet(nn.Module):
            def __init__(self):
                super(LeNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
            
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        return LeNet()
    
    def create_custom_lightweight_model(self):
        """创建自定义轻量级模型"""
        class LightweightCNN(nn.Module):
            def __init__(self):
                super(LightweightCNN, self).__init__()
                
                # 使用深度可分离卷积
                def depthwise_conv(in_ch, out_ch, stride=1):
                    return nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                        nn.BatchNorm2d(in_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )
                
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    
                    depthwise_conv(32, 64, 2),
                    depthwise_conv(64, 128, 2),
                    depthwise_conv(128, 256, 2),
                    
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.classifier = nn.Linear(256, 10)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return LightweightCNN()
    
    def create_deeper_model(self):
        """创建更深的模型"""
        class DeeperCNN(nn.Module):
            def __init__(self):
                super(DeeperCNN, self).__init__()
                
                self.features = nn.Sequential(
                    # 第一组
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # 第二组
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # 第三组
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return DeeperCNN()
    
    def train_model(self, model, model_name, epochs=10):
        """训练模型并记录性能"""
        print(f"\n开始训练 {model_name}...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 记录训练历史
        train_losses = []
        train_accuracies = []
        
        # 计算模型复杂度
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型参数数量: {total_params:,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if i % 100 == 99:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(self.train_loader)}], '
                          f'Loss: {loss.item():.4f}')
            
            # 计算epoch准确率
            epoch_acc = 100 * correct / total
            epoch_loss = running_loss / len(self.train_loader)
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}] 完成, 训练准确率: {epoch_acc:.2f}%')
        
        training_time = time.time() - start_time
        
        # 测试模型
        test_acc = self.test_model(model)
        
        # 存储结果
        self.experiment_results[model_name] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_acc,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'training_time': training_time
        }
        
        print(f"{model_name} 训练完成!")
        print(f"训练时间: {training_time:.2f}秒")
        print(f"测试准确率: {test_acc:.2f}%")
        
        return model
    
    def test_model(self, model):
        """测试模型性能"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def visualize_feature_maps(self, model, input_image):
        """可视化特征图"""
        model.eval()
        
        # 注册钩子函数来捕获中间层输出
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # 为卷积层注册钩子
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(get_activation(name))
                layer_names.append(name)
        
        # 前向传播
        with torch.no_grad():
            _ = model(input_image.unsqueeze(0).to(self.device))
        
        # 可视化前几层的特征图
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, layer_name in enumerate(layer_names[:8]):
            if layer_name in activations:
                # 取第一个样本的前几个通道
                feature_map = activations[layer_name][0]
                
                # 显示第一个通道
                if len(feature_map.shape) == 3:
                    img = feature_map[0].cpu().numpy()
                else:
                    img = feature_map.cpu().numpy()
                
                axes[i].imshow(img, cmap='viridis')
                axes[i].set_title(f'{layer_name}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compare_all_models(self):
        """对比所有模型的性能"""
        print("=== 模型架构对比实验 ===")
        
        # 创建模型
        models = {
            'LeNet': self.create_lenet_model(),
            'Lightweight': self.create_custom_lightweight_model(),
            'Deeper CNN': self.create_deeper_model()
        }
        
        # 训练所有模型
        trained_models = {}
        for name, model in models.items():
            trained_models[name] = self.train_model(model, name, epochs=5)
        
        # 可视化结果
        self.plot_comparison_results()
        
        # 特征图可视化
        sample_image, _ = next(iter(self.test_loader))
        for name, model in trained_models.items():
            print(f"\n{name} 特征图可视化:")
            self.visualize_feature_maps(model, sample_image[0])
        
        return self.experiment_results
    
    def plot_comparison_results(self):
        """绘制对比结果"""
        if not self.experiment_results:
            print("没有实验结果可显示")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 测试准确率对比
        models = list(self.experiment_results.keys())
        test_accs = [self.experiment_results[m]['test_accuracy'] for m in models]
        
        ax1.bar(models, test_accs, color=['blue', 'green', 'red'])
        ax1.set_title('测试准确率对比')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        # 参数数量对比
        params = [self.experiment_results[m]['total_params'] for m in models]
        ax2.bar(models, params, color=['blue', 'green', 'red'])
        ax2.set_title('模型参数数量对比')
        ax2.set_ylabel('参数数量')
        ax2.set_yscale('log')
        
        # 训练时间对比
        training_times = [self.experiment_results[m]['training_time'] for m in models]
        ax3.bar(models, training_times, color=['blue', 'green', 'red'])
        ax3.set_title('训练时间对比')
        ax3.set_ylabel('时间 (秒)')
        
        # 效率对比（准确率/参数数量）
        efficiency = [test_accs[i] / (params[i] / 1000) for i in range(len(models))]
        ax4.bar(models, efficiency, color=['blue', 'green', 'red'])
        ax4.set_title('效率对比 (准确率/千参数)')
        ax4.set_ylabel('效率')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细对比表
        print(f"\n{'模型':<12} {'测试准确率':<10} {'参数数量':<12} {'训练时间':<10} {'效率':<10}")
        print("-" * 70)
        for i, model in enumerate(models):
            print(f"{model:<12} {test_accs[i]:<10.2f} {params[i]:<12,} "
                  f"{training_times[i]:<10.1f} {efficiency[i]:<10.3f}")

# 使用示例
if __name__ == "__main__":
    comparator = CNNArchitectureComparator()
    results = comparator.compare_all_models()
```

</details>

---

## 练习2：高级迁移学习策略实验

**任务描述**：

+ 实现多种迁移学习策略，包括特征提取、微调、渐进式解冻等
+ 在自定义数据集上比较不同策略的效果
+ 研究不同层的学习率设置对迁移学习效果的影响
+ 实现知识蒸馏技术，将大模型的知识迁移到小模型

**提示**：

+ 使用不同的预训练模型（ResNet、VGG、EfficientNet等）作为特征提取器
+ 实现学习率调度策略，如余弦退火、指数衰减等
+ 使用验证集监控过拟合，实现早停机制
+ 可视化不同层的激活分布，理解迁移学习过程

<details>
<summary>点击查看答案</summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import copy

class AdvancedTransferLearning:
    def __init__(self, device=None):
        """高级迁移学习实验器"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 实验结果存储
        self.experiment_results = {}
    
    def create_custom_dataset(self, num_samples_per_class=200, num_classes=5):
        """创建自定义数据集"""
        class CustomDataset(Dataset):
            def __init__(self, num_samples_per_class, num_classes, transform=None, seed=42):
                torch.manual_seed(seed)
                self.transform = transform
                
                self.images = []
                self.labels = []
                
                for class_id in range(num_classes):
                    for _ in range(num_samples_per_class):
                        # 生成具有类别特征的模拟图像
                        base_image = torch.randn(3, 224, 224)
                        
                        # 为不同类别添加不同的模式
                        if class_id == 0:
                            base_image[:, 50:150, 50:150] += 1.0
                        elif class_id == 1:
                            base_image[:, 100:200, 100:200] += 0.8
                        elif class_id == 2:
                            base_image[:, :100, :100] += 0.6
                        elif class_id == 3:
                            base_image[:, 150:, 150:] += 0.4
                        else:
                            base_image[:, 75:175, 75:175] += 0.2
                        
                        self.images.append(base_image)
                        self.labels.append(class_id)
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                image = self.images[idx]
                label = self.labels[idx]
                
                # 转换为PIL图像以应用transforms
                image = transforms.ToPILImage()(image)
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        # 数据变换
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = CustomDataset(num_samples_per_class, num_classes, train_transform)
        val_dataset = CustomDataset(num_samples_per_class//4, num_classes, val_transform, seed=43)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def create_model_with_strategy(self, model_name, num_classes, strategy):
        """根据策略创建模型"""
        # 加载预训练模型
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model = model.to(self.device)
        
        # 应用迁移学习策略
        if strategy == 'feature_extraction':
            # 冻结所有特征层
            for name, param in model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
                    
        elif strategy == 'fine_tuning':
            # 所有层都可训练，但使用较小学习率
            pass
            
        elif strategy == 'gradual_unfreezing':
            # 初始时只训练分类器，后续逐步解冻
            for name, param in model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
        
        return model
    
    def setup_optimizer_with_different_lr(self, model, strategy, base_lr=0.001):
        """设置不同层使用不同学习率的优化器"""
        if strategy == 'feature_extraction':
            # 只优化分类器
            params_to_optimize = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_optimize.append(param)
            optimizer = optim.Adam(params_to_optimize, lr=base_lr)
            
        elif strategy == 'fine_tuning':
            # 特征层使用较小学习率，分类器使用正常学习率
            feature_params = []
            classifier_params = []
            
            for name, param in model.named_parameters():
                if 'fc' in name or 'classifier' in name:
                    classifier_params.append(param)
                else:
                    feature_params.append(param)
            
            optimizer = optim.Adam([
                {'params': feature_params, 'lr': base_lr * 0.1},
                {'params': classifier_params, 'lr': base_lr}
            ])
            
        else:
            # 默认优化器
            optimizer = optim.Adam(model.parameters(), lr=base_lr)
        
        return optimizer
    
    def progressive_unfreezing_train(self, model, train_loader, val_loader, epochs=15):
        """渐进式解冻训练"""
        print("开始渐进式解冻训练...")
        
        criterion = nn.CrossEntropyLoss()
        
        # 第一阶段：只训练分类器
        print("第一阶段：训练分类器")
        optimizer = self.setup_optimizer_with_different_lr(model, 'feature_extraction')
        
        stage1_results = self.train_epochs(model, train_loader, val_loader, 
                                         criterion, optimizer, epochs//3)
        
        # 第二阶段：解冻后面几层
        print("第二阶段：解冻后层")
        layer_names = [name for name, _ in model.named_parameters()]
        unfreeze_from = len(layer_names) * 2 // 3
        
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= unfreeze_from:
                param.requires_grad = True
        
        optimizer = self.setup_optimizer_with_different_lr(model, 'fine_tuning', 0.0001)
        stage2_results = self.train_epochs(model, train_loader, val_loader, 
                                         criterion, optimizer, epochs//3)
        
        # 第三阶段：全模型微调
        print("第三阶段：全模型微调")
        for param in model.parameters():
            param.requires_grad = True
            
        optimizer = self.setup_optimizer_with_different_lr(model, 'fine_tuning', 0.00001)
        stage3_results = self.train_epochs(model, train_loader, val_loader, 
                                         criterion, optimizer, epochs//3)
        
        # 合并结果
        combined_results = {
            'train_losses': stage1_results['train_losses'] + stage2_results['train_losses'] + stage3_results['train_losses'],
            'val_losses': stage1_results['val_losses'] + stage2_results['val_losses'] + stage3_results['val_losses'],
            'val_accuracies': stage1_results['val_accuracies'] + stage2_results['val_accuracies'] + stage3_results['val_accuracies']
        }
        
        return model, combined_results
    
    def train_epochs(self, model, train_loader, val_loader, criterion, optimizer, epochs):
        """训练指定轮数"""
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def knowledge_distillation(self, teacher_model, student_model, train_loader, val_loader, 
                             epochs=10, temperature=4, alpha=0.7):
        """知识蒸馏训练"""
        print("开始知识蒸馏训练...")
        
        teacher_model.eval()  # 教师模型固定
        student_model.train()
        
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 学生模型前向传播
                student_outputs = student_model(inputs)
                
                # 教师模型前向传播
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                # 计算损失
                # 硬标签损失
                loss_ce = criterion_ce(student_outputs, labels)
                
                # 软标签损失（知识蒸馏）
                loss_kd = criterion_kd(
                    torch.log_softmax(student_outputs / temperature, dim=1),
                    torch.softmax(teacher_outputs / temperature, dim=1)
                ) * (temperature ** 2)
                
                # 总损失
                loss = alpha * loss_kd + (1 - alpha) * loss_ce
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 验证
            val_acc = self.evaluate_model(student_model, val_loader)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
    
    def evaluate_model(self, model, data_loader):
        """评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def run_all_experiments(self):
        """运行所有迁移学习实验"""
        print("=== 高级迁移学习策略对比实验 ===")
        
        # 创建数据集
        train_loader, val_loader = self.create_custom_dataset()
        
        strategies = ['feature_extraction', 'fine_tuning', 'gradual_unfreezing']
        
        for strategy in strategies:
            print(f"\n实验：{strategy}")
            
            # 创建模型
            model = self.create_model_with_strategy('resnet18', 5, strategy)
            
            if strategy == 'gradual_unfreezing':
                # 渐进式解冻
                trained_model, results = self.progressive_unfreezing_train(
                    model, train_loader, val_loader, epochs=15)
            else:
                # 常规训练
                criterion = nn.CrossEntropyLoss()
                optimizer = self.setup_optimizer_with_different_lr(model, strategy)
                results = self.train_epochs(model, train_loader, val_loader, 
                                          criterion, optimizer, epochs=10)
                trained_model = model
            
            # 最终评估
            final_accuracy = self.evaluate_model(trained_model, val_loader)
            results['final_accuracy'] = final_accuracy
            
            self.experiment_results[strategy] = results
            
            print(f"{strategy} 最终准确率: {final_accuracy:.2f}%")
        
        # 知识蒸馏实验
        print("\n实验：知识蒸馏")
        teacher_model = self.create_model_with_strategy('resnet50', 5, 'fine_tuning')
        student_model = self.create_model_with_strategy('resnet18', 5, 'fine_tuning')
        
        # 先训练教师模型
        criterion = nn.CrossEntropyLoss()
        teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
        self.train_epochs(teacher_model, train_loader, val_loader, 
                         criterion, teacher_optimizer, epochs=5)
        
        # 知识蒸馏
        kd_results = self.knowledge_distillation(teacher_model, student_model, 
                                                train_loader, val_loader)
        final_kd_accuracy = self.evaluate_model(student_model, val_loader)
        kd_results['final_accuracy'] = final_kd_accuracy
        
        self.experiment_results['knowledge_distillation'] = kd_results
        
        # 可视化结果
        self.plot_experiment_results()
        
        return self.experiment_results
    
    def plot_experiment_results(self):
        """可视化实验结果"""
        if not self.experiment_results:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 最终准确率对比
        strategies = list(self.experiment_results.keys())
        final_accs = [self.experiment_results[s]['final_accuracy'] for s in strategies]
        
        colors = ['blue', 'green', 'red', 'orange']
        ax1.bar(strategies, final_accs, color=colors[:len(strategies)])
        ax1.set_title('不同策略最终准确率对比')
        ax1.set_ylabel('准确率 (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 训练损失曲线
        for i, strategy in enumerate(strategies):
            if 'train_losses' in self.experiment_results[strategy]:
                ax2.plot(self.experiment_results[strategy]['train_losses'], 
                        label=strategy, color=colors[i])
        ax2.set_title('训练损失对比')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        # 验证准确率曲线
        for i, strategy in enumerate(strategies):
            if 'val_accuracies' in self.experiment_results[strategy]:
                ax3.plot(self.experiment_results[strategy]['val_accuracies'], 
                        label=strategy, color=colors[i])
        ax3.set_title('验证准确率对比')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('准确率 (%)')
        ax3.legend()
        
        # 策略效果总结
        ax4.axis('off')
        summary_text = "策略总结:\n\n"
        for strategy in strategies:
            acc = self.experiment_results[strategy]['final_accuracy']
            summary_text += f"{strategy}: {acc:.1f}%\n"
        
        ax4.text(0.1, 0.7, summary_text, fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    experimenter = AdvancedTransferLearning()
    results = experimenter.run_all_experiments()
    
    print("\n=== 实验总结 ===")
    for strategy, result in results.items():
        print(f"{strategy}: {result['final_accuracy']:.2f}%")
```

</details>
