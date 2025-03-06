# 课后拓展参考答案

## 练习1：基础边缘检测与优化

**任务描述：**

+ 使用 Canny 算法对提供的图像进行边缘检测
+ 尝试不同的高斯滤波参数和 Canny 阈值，观察效果差异
+ 使用形态学操作（腐蚀和膨胀）优化边缘检测结果

**提示：**：

+ 尝试不同的 Canny 阈值组合（如 50-150，100-200）
+ 使用不同大小的高斯滤波核（3×3，5×5）
+ 观察并记录参数变化对边缘检测效果的影响

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

# 读取图像并转为灰度图
img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 尝试不同参数的高斯滤波
blur_3x3 = cv2.GaussianBlur(img_gray, (3, 3), 0)
blur_5x5 = cv2.GaussianBlur(img_gray, (5, 5), 0)

# 尝试不同阈值的 Canny 边缘检测
edges1 = cv2.Canny(blur_3x3, 50, 150)
edges2 = cv2.Canny(blur_3x3, 100, 200)
edges3 = cv2.Canny(blur_5x5, 50, 150)
edges4 = cv2.Canny(blur_5x5, 100, 200)

# 应用形态学操作优化边缘
kernel = np.ones((3, 3), np.uint8)
edges_eroded = cv2.erode(edges1, kernel, iterations=1)
edges_dilated = cv2.dilate(edges1, kernel, iterations=1)

# 显示结果
plt.figure(figsize=(15, 10))
images = [img_gray, blur_3x3, blur_5x5, edges1, edges2, edges3, edges4, edges_eroded, edges_dilated]
titles = ['原始灰度图', '高斯模糊 (3x3)', '高斯模糊 (5x5)', 
          'Canny (3x3, 50-150)', 'Canny (3x3, 100-200)', 
          'Canny (5x5, 50-150)', 'Canny (5x5, 100-200)',
          '腐蚀后的边缘', '膨胀后的边缘']

for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
```

</details>

---

## 练习2：形态学操作实践

**任务描述：**

+ 对边缘检测结果应用不同的形态学操作（腐蚀、膨胀、开运算、闭运算）
+ 比较不同操作对去除噪点和连接断开边缘的效果
+ 尝试不同大小和形状的结构元素（矩形、椭圆形、十字形）

**提示：**

+ 创建 3×3 和 5×5 大小的不同形状结构元素
+ 应用单次和多次迭代的形态学操作
+ 特别关注开运算对去噪和闭运算对连接断开边缘的效果

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

# 读取图像并检测边缘
img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
edges = cv2.Canny(img_blur, 20, 60)

# 创建不同的结构元素
rect_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
rect_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
ellipse_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cross_3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# 应用不同的形态学操作
erosion = cv2.erode(edges, rect_3x3, iterations=1)
dilation = cv2.dilate(edges, rect_3x3, iterations=1)
opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, rect_3x3)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect_3x3)

# 使用不同结构元素的开运算
opening_rect = cv2.morphologyEx(edges, cv2.MORPH_OPEN, rect_3x3)
opening_ellipse = cv2.morphologyEx(edges, cv2.MORPH_OPEN, ellipse_3x3)
opening_cross = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cross_3x3)

# 使用不同结构元素的闭运算
closing_rect = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect_3x3)
closing_ellipse = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ellipse_3x3)
closing_cross = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cross_3x3)

# 显示结果
plt.figure(figsize=(15, 10))
images = [edges, erosion, dilation, opening, closing,
          opening_rect, opening_ellipse, opening_cross,
          closing_rect, closing_ellipse, closing_cross]
titles = ['原始边缘', '腐蚀', '膨胀', '开运算', '闭运算',
          '矩形开运算', '椭圆开运算', '十字开运算',
          '矩形闭运算', '椭圆闭运算', '十字闭运算']

for i in range(len(images)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
```

</details>

---

## 练习3：边缘提取应用

**任务描述：**

+ 设计一个简单的边缘提取流程，包括预处理、边缘检测和形态学优化
+ 处理不同类型的图像（如风景、物体、文本）
+ 为每种图像类型找到最适合的处理参数

**提示：**

+ 遵循课程中介绍的处理流程：预处理→边缘检测→形态学操作
+ 对比 Sobel、Laplacian 和 Canny 算法的边缘检测效果
+ 通过实验找到每种图像类型的最佳处理参数组合

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

def process_image(image_path, blur_size=5, canny_low=20, canny_high=60, operation='close', kernel_size=3):
    """完整的边缘提取流程"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
        
    # 转换为灰度图并去噪
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)
    
    # 边缘检测
    # Sobel 算子
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel / sobel.max() * 255)
    
    # Laplacian 算子
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian) / laplacian.max() * 255)
    
    # Canny 算子
    canny = cv2.Canny(img_blur, canny_low, canny_high)
    
    # 选择要优化的边缘
    edges = canny  # 这里使用 Canny 算法的结果
    
    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 形态学操作
    if operation == 'erode':
        result = cv2.erode(edges, kernel, iterations=1)
    elif operation == 'dilate':
        result = cv2.dilate(edges, kernel, iterations=1)
    elif operation == 'open':
        result = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    else:
        result = edges
    
    # 返回处理结果和中间结果
    return {
        'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'gray': img_gray,
        'blur': img_blur,
        'sobel': sobel,
        'laplacian': laplacian,
        'canny': canny,
        'result': result
    }

# 处理图像示例
image_path = 'image.jpg'  # 替换为你的图像路径
results = process_image(image_path)

# 显示结果
plt.figure(figsize=(15, 10))
images = [results['original'], results['gray'], results['blur'], 
          results['sobel'], results['laplacian'], results['canny'], results['result']]
titles = ['原始图像', '灰度图', '高斯模糊', 
          'Sobel 边缘', 'Laplacian 边缘', 'Canny 边缘', '最终结果']

for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    if i == 0:  # 原始彩色图像
        plt.imshow(images[i])
    else:  # 灰度图像
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# 尝试不同参数组合
params = [
    {'blur_size': 3, 'canny_low': 50, 'canny_high': 150, 'operation': 'close', 'kernel_size': 3},
    {'blur_size': 5, 'canny_low': 100, 'canny_high': 200, 'operation': 'open', 'kernel_size': 3},
    {'blur_size': 5, 'canny_low': 30, 'canny_high': 100, 'operation': 'dilate', 'kernel_size': 3}
]

# 显示不同参数的效果（可选）
plt.figure(figsize=(15, 5))
for i, param in enumerate(params):
    result = process_image(image_path, **param)['result']
    plt.subplot(1, 3, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(f"参数组合 {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

</details>
