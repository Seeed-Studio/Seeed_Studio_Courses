# 课后拓展参考答案

## 练习1：图像增强器

**任务描述：**

+ 开发一个图像增强工具，实现以下功能：
  + 应用不同的滤波方法处理图像
  + 对比并评估不同处理方法的效果
+ 要求处理过程可视化，便于观察效果

**提示：**：

+ 使用 OpenCV 的图像处理函数
+ 通过 Matplotlib 实现可视化
+ 注意处理参数的选择和优化

<details>
<summary>点击查看答案</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class ImageEnhancer:
    """图像增强器"""
    
    def __init__(self):
        self.image = None
        
    def apply_filters(self, image):
        """应用不同的滤波器"""
        # 高斯滤波
        gaussian = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 中值滤波
        median = cv2.medianBlur(image, 5)
        
        # 双边滤波
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        return {
            '高斯滤波': gaussian,
            '中值滤波': median,
            '双边滤波': bilateral
        }
    
    def enhance_image(self, image_path):
        """增强图像并显示结果"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像")
            return
            
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用滤波器
        filtered = self.apply_filters(image)
        
        # 显示结果
        plt.figure(figsize=(10, 8))
        
        # 2x2布局
        images = {
            '原始图像': image,
            '高斯滤波': filtered['高斯滤波'],
            '中值滤波': filtered['中值滤波'],
            '双边滤波': filtered['双边滤波']
        }
        
        for idx, (title, img) in enumerate(images.items(), 1):
            plt.subplot(2, 2, idx)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# 使用示例
enhancer = ImageEnhancer()
enhancer.enhance_image('image.jpg')

```

</details>

---

## 练习2：图像几何变换工具

**任务描述：**

+ 实现一个图像几何变换工具，包括：
  + 图像的缩放功能（支持不同的插值方法）
  + 图像的旋转功能（支持任意角度旋转）
  + 图像的翻转功能（支持水平和垂直翻转）

**提示：**

+ 使用 OpenCV 的几何变换函数
+ 注意比较不同插值方法的效果
+ 处理图像旋转时注意边界的填充

<details>
<summary>点击查看答案</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class ImageTransformer:
    """图像几何变换工具"""
    
    def __init__(self):
        self.image = None
        
    def load_image(self, image_path):
        """加载图像"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("无法读取图像")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.image
    
    def resize_image(self, scale=0.5, method=cv2.INTER_LINEAR):
        """调整图像大小
        
        参数：
            scale: 缩放比例
            method: 插值方法
        """
        height, width = self.image.shape[:2]
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(self.image, new_size, interpolation=method)
    
    def rotate_image(self, angle):
        """旋转图像
        
        参数：
            angle: 旋转角度
        """
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(self.image, matrix, (width, height))
    
    def flip_image(self, direction='horizontal'):
        """翻转图像
        
        参数：
            direction: 翻转方向，'horizontal' 或 'vertical'
        """
        if direction == 'horizontal':
            return cv2.flip(self.image, 1)
        else:
            return cv2.flip(self.image, 0)
    
    def show_results(self):
        """显示所有变换效果"""
        # 应用不同的变换
        resized = self.resize_image(0.5)
        rotated = self.rotate_image(45)
        flipped_h = self.flip_image('horizontal')
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 显示原始图像
        plt.subplot(221)
        plt.imshow(self.image)
        plt.title('原始图像')
        plt.axis('off')
        
        # 显示缩放结果
        plt.subplot(222)
        plt.imshow(resized)
        plt.title('缩放效果 (0.5倍)')
        plt.axis('off')
        
        # 显示旋转结果
        plt.subplot(223)
        plt.imshow(rotated)
        plt.title('旋转效果 (45度)')
        plt.axis('off')
        
        # 显示翻转结果
        plt.subplot(224)
        plt.imshow(flipped_h)
        plt.title('水平翻转效果')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# 使用示例
transformer = ImageTransformer()
transformer.load_image('image.jpg')
transformer.show_results()
```

</details>

---

## 练习3：阈值化方法比较实验

**任务描述：**

+ 对比不同阈值化方法的效果：
  + 实现全局阈值化、自适应阈值化和 Otsu 方法
  + 分析不同方法在各种场景下的表现
  + 总结每种方法的适用条件

**提示：**

+ 使用不同类型的图像进行测试
+ 记录和比较处理结果
+ 撰写分析报告

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
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class ThresholdingAnalyzer:
    """阈值化方法分析器"""
    
    def __init__(self):
        self.image = None
        self.results = {}
        
    def apply_thresholding(self, image):
        """应用不同的阈值化方法"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 全局阈值化
        _, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 自适应阈值化 - 均值
        adaptive_mean = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # 自适应阈值化 - 高斯
        adaptive_gaussian = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Otsu阈值化
        _, otsu_thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return {
            'gray': gray,
            'global': global_thresh,
            'adaptive_mean': adaptive_mean,
            'adaptive_gaussian': adaptive_gaussian,
            'otsu': otsu_thresh
        }
    
    def analyze_image(self, image_path):
        """分析图像并显示结果"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像")
            return
            
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用阈值化
        results = self.apply_thresholding(image)
        
        # 显示结果
        plt.figure(figsize=(15, 10))
        
        titles = {
            'gray': '灰度图像',
            'global': '全局阈值化',
            'adaptive_mean': '自适应阈值化(均值)',
            'adaptive_gaussian': '自适应阈值化(高斯)',
            'otsu': 'Otsu阈值化'
        }
        
        for idx, (key, title) in enumerate(titles.items(), 1):
            plt.subplot(2, 3, idx)
            plt.imshow(results[key], cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        # 添加直方图
        plt.subplot(236)
        plt.hist(results['gray'].ravel(), 256, [0, 256])
        plt.title('灰度直方图')
        plt.xlabel('像素值')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.show()
        
        # 比较不同方法的结果
        self.compare_methods(results)
    
    def compare_methods(self, results):
        """比较不同阈值化方法的效果"""
        # 计算每种方法的像素分布
        stats = {}
        for method, result in results.items():
            if method != 'gray':
                white_pixels = np.sum(result == 255)
                black_pixels = np.sum(result == 0)
                ratio = white_pixels / (white_pixels + black_pixels)
                stats[method] = {
                    'white_ratio': ratio,
                    'black_ratio': 1 - ratio
                }
        
        # 显示统计结果
        print("\n各方法效果比较：")
        for method, stat in stats.items():
            print(f"\n{method}:")
            print(f"  白色像素占比: {stat['white_ratio']:.2%}")
            print(f"  黑色像素占比: {stat['black_ratio']:.2%}")

# 使用示例
analyzer = ThresholdingAnalyzer()
analyzer.analyze_image('test_image.jpg')
```

</details>
