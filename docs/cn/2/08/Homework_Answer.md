# 课后拓展参考答案

## 练习1：图像预处理工具

**任务描述：**

+ 开发一个图像预处理工具，实现以下功能：
  + 支持批量读取图像文件
  + 调整图像大小到指定分辨率
  + 转换颜色空间（RGB、灰度）
  + 保存处理后的图像
+ 要求能处理常见图像格式（jpg、png等）
+ 实现友好的进度显示

**提示：**

+ 使用 OpenCV 进行图像读取和处理
+ 通过 Matplotlib 显示处理效果
+ 注意处理图像读取和保存时的异常

<details>
<summary>点击查看答案</summary>

```python
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

# 设置字体为中文支持的字体，如 SimHei（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ImagePreprocessor:
    def __init__(self, input_dir, output_dir, target_size=(224, 224)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        os.makedirs(output_dir, exist_ok=True)
        
    def process_images(self):
        """批量处理图像"""
        image_files = [f for f in os.listdir(self.input_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc="处理图像"):
            try:
                # 读取图像
                image_path = os.path.join(self.input_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"无法读取图像：{image_file}")
                    continue
                
                # 调整大小
                resized = cv2.resize(image, self.target_size)
                
                # 转换为灰度图
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                
                # 保存处理后的图像
                output_path = os.path.join(self.output_dir, f"processed_{image_file}")
                cv2.imwrite(output_path, gray)
                
                # 显示原始图像与处理后的图像
                self.show_comparison(image, gray, image_file)
                
            except Exception as e:
                print(f"处理图像 {image_file} 时出错：{str(e)}")
    
    def show_comparison(self, original, processed, image_name):
        """显示处理前与处理后的图像并排"""
        # 创建一个 1x2 的子图
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原始图像
        ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # 转换为 RGB 显示
        ax[0].set_title(f'{image_name} - 原始图像')
        ax[0].axis('off')  # 不显示坐标轴
        
        # 显示处理后的图像
        if len(processed.shape) == 2:  # 如果是灰度图，直接显示
            ax[1].imshow(processed, cmap='gray')
        else:
            ax[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))  # 转换为 RGB 显示
        ax[1].set_title(f'{image_name} - 处理后的图像')
        ax[1].axis('off')  # 不显示坐标轴
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    processor = ImagePreprocessor("input_images", "output_images")
    processor.process_images()
```

</details>

---

## 练习2：图像分析器

**任务描述：**

+ 创建一个图像分析工具，实现：
  + 统计图像的亮度分布
  + 分析RGB通道的特征
  + 生成图像质量报告
  + 可视化分析结果
+ 支持批量处理多个图像
+ 生成分析报告

**提示：**

+ 使用 NumPy 进行统计分析
+ 用 Matplotlib 创建可视化图表
+ 合理组织和展示分析结果
+ 处理异常图像情况

<details>
<summary>点击查看答案</summary>

```python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.widgets import Slider
import matplotlib

# 设置字体为中文支持的字体，如 SimHei（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ImageAnalyzer:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def analyze_image(self, image_path):
        """分析单张图像的亮度分布与RGB通道特征"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像：{image_path}")
        
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 计算亮度（灰度图像）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 亮度分布统计
        brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 计算RGB通道的均值和标准差
        mean_r, mean_g, mean_b = np.mean(image_rgb, axis=(0, 1))
        std_r, std_g, std_b = np.std(image_rgb, axis=(0, 1))
        
        return brightness_hist, (mean_r, mean_g, mean_b), (std_r, std_g, std_b)
    
    def analyze_images(self):
        """批量分析图像"""
        report = {}
        
        for image_file in tqdm(self.image_files, desc="分析图像"):
            try:
                image_path = os.path.join(self.input_dir, image_file)
                brightness_hist, mean_rgb, std_rgb = self.analyze_image(image_path)
                
                # 生成报告
                report[image_file] = {
                    '亮度分布': brightness_hist,
                    'RGB均值': mean_rgb,
                    'RGB标准差': std_rgb
                }
                
                # 可视化结果
                self.plot_histogram(brightness_hist, image_file)
                self.plot_rgb_stats(mean_rgb, std_rgb, image_file)
                
            except Exception as e:
                print(f"分析图像 {image_file} 时出错：{str(e)}")
        
        return report
    
    def plot_histogram(self, brightness_hist, image_name):
        """绘制亮度分布直方图"""
        plt.figure(figsize=(8, 6))
        plt.plot(brightness_hist)
        plt.title(f'{image_name} 亮度分布')
        plt.xlabel('亮度值')
        plt.ylabel('像素数量')
        plt.grid(True)
        plt.savefig(f'brightness_histogram_{image_name}.png')
        plt.close()
    
    def plot_rgb_stats(self, mean_rgb, std_rgb, image_name):
        """绘制RGB均值和标准差"""
        labels = ['R', 'G', 'B']
        means = mean_rgb
        stds = std_rgb
        
        # 绘制均值和标准差
        x = np.arange(3)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width/2, means, width, label='均值', color='lightblue')
        ax.bar(x + width/2, stds, width, label='标准差', color='salmon')
        
        ax.set_ylabel('值')
        ax.set_title(f'{image_name} RGB 通道统计')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.savefig(f'rgb_stats_{image_name}.png')
        plt.close()

# 使用示例
if __name__ == "__main__":
    analyzer = ImageAnalyzer('input_images')  # 输入图像目录
    report = analyzer.analyze_images()
    
    # 保存分析报告
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        for image_file, stats in report.items():
            f.write(f"图像：{image_file}\n")
            f.write(f"RGB均值：{stats['RGB均值']}\n")
            f.write(f"RGB标准差：{stats['RGB标准差']}\n")
            f.write("="*50 + "\n")
    
    print("图像分析完成，报告已生成。")
```

</details>

---

## 练习3：图像增强工具

**任务描述：**

+ 开发一个图像增强工具，包含：
  + 亮度和对比度调整
  + 颜色平衡优化
  + 图像锐化处理
  + 降噪处理
+ 实现实时预览效果
+ 支持参数调整和效果比较

**提示：**

+ 使用 OpenCV 的图像处理函数
+ 通过 Matplotlib 实现交互式显示
+ 注意处理参数范围和边界情况
+ 保持原始图像以便比较

<details>
<summary>点击查看答案</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib

# 设置字体为中文支持的字体，如 SimHei（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ImageEnhancer:
    def __init__(self, image_path):
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"无法读取图像：{image_path}")
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.current = self.original.copy()
        
    def adjust_brightness_contrast(self, brightness=0, contrast=1):
        """调整亮度和对比度"""
        self.current = cv2.convertScaleAbs(
            self.original, 
            alpha=contrast, 
            beta=brightness
        )
        
    def sharpen_image(self, amount=1.0):
        """图像锐化"""
        kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]]) * amount
        self.current = cv2.filter2D(self.current, -1, kernel)
        
    def reduce_noise(self, strength=5):
        """降噪处理"""
        self.current = cv2.GaussianBlur(
            self.current, 
            (strength, strength), 
            0
        )
        
    def interactive_enhance(self):
        """交互式图像增强"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # 显示原始图像和处理后的图像
        ax1.imshow(self.original)
        ax1.set_title('原始图像')
        ax1.axis('off')
        
        img_enhanced = ax2.imshow(self.current)
        ax2.set_title('增强后的图像')
        ax2.axis('off')
        
        # 创建滑动条
        axbrightness = plt.axes([0.2, 0.1, 0.6, 0.03])
        axcontrast = plt.axes([0.2, 0.15, 0.6, 0.03])
        
        brightness_slider = Slider(axbrightness, '亮度', -100, 100, valinit=0)
        contrast_slider = Slider(axcontrast, '对比度', 0.1, 3.0, valinit=1.0)
        
        def update(val):
            brightness = brightness_slider.val
            contrast = contrast_slider.val
            self.adjust_brightness_contrast(brightness, contrast)
            img_enhanced.set_array(self.current)
            fig.canvas.draw_idle()
            
        brightness_slider.on_changed(update)
        contrast_slider.on_changed(update)
        
        plt.show()

# 使用示例
if __name__ == "__main__":
    enhancer = ImageEnhancer("image.jpg")
    enhancer.interactive_enhance()
```

</details>
