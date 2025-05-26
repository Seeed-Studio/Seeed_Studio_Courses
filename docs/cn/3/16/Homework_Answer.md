# 课后拓展参考答案

## 练习1：Tensor 操作与自动求导综合练习

**任务描述**：

+ 构建一个 Tensor 操作练习程序，通过实际计算加深对 PyTorch 核心数据结构的理解
+ 实现手动的矩阵乘法运算，模拟神经网络中的线性变换过程
+ 对比不同 Tensor 创建方法的性能差异，理解内存管理的重要性
+ 使用自动求导功能计算简单函数的梯度，验证手动推导的结果

**提示**：

+ 使用 `torch.randn()`, `torch.zeros()`, `torch.ones()` 等方法创建不同类型的 Tensor
+ 利用 `torch.mm()` 或 `@` 操作符进行矩阵乘法运算
+ 通过 `requires_grad=True` 参数启用梯度跟踪，使用 `backward()` 计算梯度
+ 使用 `torch.cuda.is_available()` 检查 GPU 可用性，对比 CPU 和 GPU 上的计算性能

<details>
<summary>点击查看答案</summary>

```python
import torch
import time
import matplotlib.pyplot as plt

class TensorExerciser:
    def __init__(self):
        """初始化 Tensor 练习器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def creation_benchmark(self, size=(1000, 1000)):
        """对比不同 Tensor 创建方法的性能"""
        print(f"\n=== Tensor 创建方法性能对比 (大小: {size}) ===")
        
        methods = {
            'zeros': lambda: torch.zeros(size, device=self.device),
            'ones': lambda: torch.ones(size, device=self.device),
            'randn': lambda: torch.randn(size, device=self.device),
            'empty': lambda: torch.empty(size, device=self.device)
        }
        
        for name, method in methods.items():
            start_time = time.time()
            tensor = method()
            end_time = time.time()
            
            print(f"{name:10}: {end_time - start_time:.4f}秒, 内存使用: {tensor.element_size() * tensor.nelement() / 1024**2:.2f}MB")
    
    def linear_transformation_demo(self):
        """演示线性变换在神经网络中的应用"""
        print(f"\n=== 线性变换演示 ===")
        
        # 模拟一个简单的全连接层
        input_features = 4
        output_features = 3
        batch_size = 2
        
        # 创建输入数据
        inputs = torch.randn(batch_size, input_features, device=self.device)
        print(f"输入数据形状: {inputs.shape}")
        print(f"输入数据:\n{inputs}")
        
        # 创建权重和偏置
        weights = torch.randn(input_features, output_features, device=self.device)
        bias = torch.randn(output_features, device=self.device)
        
        print(f"\n权重形状: {weights.shape}")
        print(f"偏置形状: {bias.shape}")
        
        # 执行线性变换: y = xW + b
        output = torch.mm(inputs, weights) + bias
        print(f"\n输出形状: {output.shape}")
        print(f"输出数据:\n{output}")
        
        # 展示广播机制
        print(f"\n广播机制演示:")
        print(f"inputs @ weights 的形状: {torch.mm(inputs, weights).shape}")
        print(f"bias 被广播到: {bias.expand_as(torch.mm(inputs, weights)).shape}")
    
    def autograd_demo(self):
        """演示自动求导功能"""
        print(f"\n=== 自动求导演示 ===")
        
        # 创建需要计算梯度的变量
        x = torch.tensor(2.0, requires_grad=True, device=self.device)
        y = torch.tensor(3.0, requires_grad=True, device=self.device)
        
        print(f"输入变量: x = {x.item()}, y = {y.item()}")
        
        # 定义复合函数: f(x, y) = x^2 * y + y^3
        f = x**2 * y + y**3
        print(f"函数值: f(x, y) = x^2 * y + y^3 = {f.item()}")
        
        # 自动计算梯度
        f.backward()
        
        print(f"自动计算的梯度:")
        print(f"∂f/∂x = {x.grad.item()}")  # 应该是 2*x*y = 2*2*3 = 12
        print(f"∂f/∂y = {y.grad.item()}")  # 应该是 x^2 + 3*y^2 = 4 + 27 = 31
        
        # 手动验证梯度
        print(f"\n手动计算验证:")
        print(f"∂f/∂x = 2*x*y = 2*{x.item()}*{y.item()} = {2*x.item()*y.item()}")
        print(f"∂f/∂y = x^2 + 3*y^2 = {x.item()}^2 + 3*{y.item()}^2 = {x.item()**2 + 3*y.item()**2}")
    
    def shape_manipulation_practice(self):
        """练习 Tensor 形状变换操作"""
        print(f"\n=== 形状变换练习 ===")
        
        # 创建一个 4D tensor 模拟批量图像
        batch_size, channels, height, width = 2, 3, 4, 4
        images = torch.randn(batch_size, channels, height, width, device=self.device)
        print(f"原始图像 tensor 形状: {images.shape}")
        
        # 展平操作（模拟进入全连接层前的操作）
        flattened = images.view(batch_size, -1)
        print(f"展平后形状: {flattened.shape}")
        
        # 重塑回原始形状
        reshaped = flattened.view(batch_size, channels, height, width)
        print(f"重塑回原形状: {reshaped.shape}")
        
        # 验证数据一致性
        print(f"重塑后数据是否一致: {torch.equal(images, reshaped)}")
        
        # 转置操作
        transposed = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
        print(f"转置后形状 (NCHW -> NHWC): {transposed.shape}")
    
    def run_all_exercises(self):
        """运行所有练习"""
        self.creation_benchmark()
        self.linear_transformation_demo()
        self.autograd_demo()
        self.shape_manipulation_practice()

# 使用示例
if __name__ == "__main__":
    exerciser = TensorExerciser()
    exerciser.run_all_exercises()
```

</details>

---

## 练习2：神经网络构建与训练流程实现

**任务描述**：

+ 使用 nn.Module 构建一个两层的全连接神经网络，处理简单的分类任务
+ 实现完整的训练循环，包括前向传播、损失计算、反向传播和参数更新
+ 比较不同激活函数（ReLU、Sigmoid、Tanh）对模型训练效果的影响
+ 可视化训练过程中损失函数的变化趋势，分析模型的收敛情况

**提示**：

+ 使用 `nn.Linear` 定义全连接层，`nn.ReLU()` 等定义激活函数
+ 选择合适的损失函数，如 `nn.CrossEntropyLoss` 用于分类任务
+ 使用 `torch.optim.Adam` 或 `torch.optim.SGD` 作为优化器
+ 在每个训练周期使用 `optimizer.zero_grad()` 清零梯度，避免梯度累积问题
+ 利用 `matplotlib` 绘制训练损失曲线，观察模型学习过程

<details>
<summary>点击查看答案</summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体，支持显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SimpleNetworkBuilder:
    def __init__(self, input_size, output_size):
        """
        初始化网络构建器
        
        参数:
            input_size: 输入特征数量
            output_size: 输出类别数量
        """
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def build_network(self, hidden_layers, activation='relu', dropout_prob=0.0):
        """
        构建神经网络
        
        参数:
            hidden_layers: 隐藏层神经元数量列表，如 [64, 32]
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
            dropout_prob: Dropout 概率
        """
        layers = []
        
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        
        act_fn = activation_map.get(activation, nn.ReLU())
        
        # 构建网络层
        prev_size = self.input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn)
            
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, self.output_size))
        
        # 如果是二分类，添加 Sigmoid
        if self.output_size == 1:
            layers.append(nn.Sigmoid())
        
        network = nn.Sequential(*layers)
        return network.to(self.device)
    
    def count_parameters(self, model):
        """统计模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        return total_params, trainable_params
    
    def generate_sample_data(self, num_samples=1000):
        """生成简单的二分类样本数据"""
        # 生成两个高斯分布的数据点
        np.random.seed(42)
        
        # 类别 0
        class0 = np.random.normal([2, 2], [1, 1], (num_samples//2, 2))
        labels0 = np.zeros((num_samples//2, 1))
        
        # 类别 1
        class1 = np.random.normal([-2, -2], [1, 1], (num_samples//2, 2))
        labels1 = np.ones((num_samples//2, 1))
        
        # 合并数据
        X = np.vstack([class0, class1])
        y = np.vstack([labels0, labels1])
        
        # 转换为 tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def train_and_evaluate(self, model, X, y, epochs=100, lr=0.01):
        """训练和评估模型"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        losses = []
        accuracies = []
        
        print(f"开始训练，共 {epochs} 个 epoch...")
        
        for epoch in range(epochs):
            # 训练模式
            model.train()
            
            # 前向传播
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            losses.append(loss.item())
            
            # 计算准确率
            with torch.no_grad():
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == y).float().mean().item()
                accuracies.append(accuracy)
            
            # 打印进度
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        
        return losses, accuracies
    
    def plot_training_progress(self, losses, accuracies):
        """绘制训练进度"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(losses)
        ax1.set_title('训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(accuracies)
        ax2.set_title('训练准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_configurations(self):
        """对比不同网络配置的效果"""
        print("=== 网络配置对比实验 ===")
        
        # 生成数据
        X, y = self.generate_sample_data(1000)
        
        # 不同配置
        configurations = [
            {"hidden_layers": [16], "activation": "relu", "name": "浅层网络"},
            {"hidden_layers": [32, 16], "activation": "relu", "name": "深层网络"},
            {"hidden_layers": [32, 16], "activation": "sigmoid", "name": "Sigmoid激活"},
            {"hidden_layers": [32, 16], "activation": "relu", "dropout_prob": 0.2, "name": "带Dropout"}
        ]
        
        results = []
        
        for config in configurations:
            print(f"\n--- 测试配置: {config['name']} ---")
            
            # 构建网络
            model = self.build_network(**{k: v for k, v in config.items() if k != 'name'})
            
            # 统计参数
            total_params, _ = self.count_parameters(model)
            
            # 训练
            losses, accuracies = self.train_and_evaluate(model, X, y, epochs=50)
            
            # 记录结果
            final_accuracy = accuracies[-1]
            results.append({
                'name': config['name'],
                'params': total_params,
                'final_accuracy': final_accuracy,
                'losses': losses
            })
            
            print(f"最终准确率: {final_accuracy:.4f}")
        
        # 绘制对比图
        plt.figure(figsize=(12, 8))
        
        # 损失对比
        plt.subplot(2, 2, 1)
        for result in results:
            plt.plot(result['losses'], label=result['name'])
        plt.title('不同配置的损失对比')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 参数数量对比
        plt.subplot(2, 2, 2)
        names = [r['name'] for r in results]
        params = [r['params'] for r in results]
        plt.bar(names, params)
        plt.title('参数数量对比')
        plt.ylabel('参数数量')
        plt.xticks(rotation=45)
        
        # 最终准确率对比
        plt.subplot(2, 2, 3)
        accuracies = [r['final_accuracy'] for r in results]
        plt.bar(names, accuracies)
        plt.title('最终准确率对比')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 总结对比
        print(f"\n=== 配置对比总结 ===")
        for result in results:
            print(f"{result['name']:15}: 参数数量 {result['params']:,}, 最终准确率 {result['final_accuracy']:.4f}")
        
        return results

# 使用示例
if __name__ == "__main__":
    # 创建网络构建器
    builder = SimpleNetworkBuilder(input_size=2, output_size=1)
    
    # 运行配置对比实验
    results = builder.compare_configurations()
```

</details>
