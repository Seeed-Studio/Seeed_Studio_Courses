# 课后拓展参考答案

## 练习1：神经元计算模拟器

**任务描述**：

+ 编写一个程序来模拟单个神经元的计算过程，帮助理解神经网络的基本工作原理
+ 实现一个接收3个输入特征的神经元，并使用sigmoid激活函数
+ 通过调整权重和偏置，观察神经元对不同输入的响应变化
+ 设计一个简单的猫特征识别器，展示神经元如何进行基本分类

**提示**：

+ 关注神经元的四个关键组成部分：输入特征、权重、偏置和激活函数
+ 使用numpy库进行向量计算，提高代码效率
+ 尝试不同的权重组合，观察它们对最终分类结果的影响
+ 记录每一步的中间计算结果，帮助理解神经元的信息处理流程

<details>
<summary>点击查看答案</summary>

```python
import numpy as np

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

class SimpleNeuron:
    def __init__(self, weights, bias):
        """初始化神经元"""
        self.weights = np.array(weights)
        self.bias = bias
    
    def compute(self, inputs):
        """计算神经元输出"""
        # 确保输入数量正确
        if len(inputs) != len(self.weights):
            raise ValueError("输入数量必须与权重数量相同")
        
        # 计算加权和
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        print(f"加权和: {weighted_sum:.4f}")
        
        # 通过激活函数
        output = sigmoid(weighted_sum)
        print(f"激活后输出: {output:.4f}")
        
        return output

# 使用示例
weights = [0.5, -0.2, 0.1]  # 三个输入的权重
bias = 0.2                  # 偏置值
neuron = SimpleNeuron(weights, bias)

# 测试数据
inputs = [1.0, 0.5, 0.8]
print("输入值:", inputs)
output = neuron.compute(inputs)
```

</details>

---

## 练习2：一键式图像预处理与增强工具

**任务描述**：

+ 基于课程中的`BatchImageProcessor`（预处理）和`BatchAugmenter`（数据增强）两个类，构建一个完整的图像处理管道
+ 实现从原始图像到训练就绪数据的一键式转换流程
+ 保证预处理和增强步骤的正确顺序执行，先预处理再增强
+ 支持按类别处理图像并维持目录结构

**提示**：

+ 使用`BatchImageProcessor`处理所有原始图像，标准化尺寸和像素值
+ 使用`BatchAugmenter`对预处理后的图像进行增强，扩充数据集
+ 设计合理的目录结构，区分原始、预处理和增强后的图像
+ 考虑训练集和验证集的分离，通常只对训练集进行增强

<details>
<summary>点击查看答案</summary>

```python
import cv2
import numpy as np
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# 注：假设BatchImageProcessor和BatchAugmenter类已在其他文件中定义
from image_processor import BatchImageProcessor
from image_augmenter import BatchAugmenter  

def create_dataset_structure(base_path):
    """创建数据集的标准目录结构"""
    dirs = [
        f"{base_path}/train/cat",
        f"{base_path}/train/dog",
        f"{base_path}/val/cat",
        f"{base_path}/val/dog",
        f"{base_path}/processed/train/cat",
        f"{base_path}/processed/train/dog",
        f"{base_path}/processed/val/cat",
        f"{base_path}/processed/val/dog",
        f"{base_path}/augmented/train/cat",
        f"{base_path}/augmented/train/dog"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

def process_dataset(raw_data_path, output_path, val_split=0.2):
    """
    完整的数据集处理流程：从原始图像到预处理和增强
    
    参数:
        raw_data_path: 包含cat和dog子目录的原始数据目录
        output_path: 输出目录
        val_split: 验证集比例
    """
    # 创建目录结构
    create_dataset_structure(output_path)
    
    # 处理每个类别
    for class_name in ["cat", "dog"]:
        class_dir = f"{raw_data_path}/{class_name}"
        if not os.path.exists(class_dir):
            print(f"警告: 类别目录不存在 {class_dir}")
            continue
            
        # 获取所有图像
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in valid_extensions:
            images.extend(list(Path(class_dir).glob(f"*{ext}")))
        
        if not images:
            print(f"警告: 在 {class_dir} 中没有找到图像")
            continue
            
        # 划分训练集和验证集
        train_images, val_images = train_test_split(
            images, test_size=val_split, random_state=42
        )
        
        # 复制原始图像到训练和验证目录
        for img in train_images:
            dest = f"{output_path}/train/{class_name}/{img.name}"
            shutil.copy2(img, dest)
        
        for img in val_images:
            dest = f"{output_path}/val/{class_name}/{img.name}"
            shutil.copy2(img, dest)
            
        print(f"类别 {class_name}: 分配 {len(train_images)} 张图像到训练集, {len(val_images)} 张到验证集")
        
        # 第1步: 预处理训练和验证集
        processor = BatchImageProcessor(target_size=(224, 224))
        
        processor.process_directory(
            f"{output_path}/train/{class_name}", 
            f"{output_path}/processed/train/{class_name}"
        )
        
        processor.process_directory(
            f"{output_path}/val/{class_name}", 
            f"{output_path}/processed/val/{class_name}"
        )
        
        # 第2步: 只对训练集的预处理图像进行增强
        augmenter = BatchAugmenter()
        
        augmenter.augment_directory(
            f"{output_path}/processed/train/{class_name}",
            f"{output_path}/augmented/train/{class_name}",
            augmentations_per_image=3
        )
    
    print("数据集处理完成! 结构如下:")
    print(f"- 原始分类数据: {output_path}/train/ 和 {output_path}/val/")
    print(f"- 预处理数据: {output_path}/processed/")
    print(f"- 增强数据: {output_path}/augmented/")

# 使用示例
if __name__ == "__main__":
    # 处理猫狗数据集
    process_dataset(
        raw_data_path="./raw_data",  # 包含cat和dog子目录的原始数据
        output_path="./pet_dataset",
        val_split=0.2
    )
```

</details>
