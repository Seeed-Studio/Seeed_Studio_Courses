# 课后拓展参考答案

## 练习1：轮廓分析与形状识别

**任务描述：**

+ 开发一个程序，利用轮廓检测识别图像中的基本几何形状（圆形、三角形、矩形等）
+ 根据轮廓特征（顶点数量、面积、周长等）对不同形状进行分类
+ 在图像上标注识别出的形状名称

**提示：**：

+ 使用 `cv2.findContours()` 检测二值图像中的轮廓
+ 使用 `cv2.approxPolyDP()` 近似轮廓，根据顶点数判断形状
+ 计算轮廓的面积和周长，进一步辅助形状判断

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

def detect_shapes(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像，请检查图像路径")
        return None
    
    # 创建结果图像的副本
    result = img.copy()
    
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 膨胀边缘，使轮廓更加明显
    dilated = cv2.dilate(edges, None, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"检测到 {len(contours)} 个轮廓")
    
    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓面积，忽略过小的轮廓
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 计算形状的复杂度（周长与面积的比值）
        complexity = perimeter / (2 * np.sqrt(np.pi * area))
        
        # 轮廓近似
        epsilon = 0.02 * perimeter  # 近似精度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 获取轮廓的中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # 根据顶点数量和形状复杂度判断形状
        vertices = len(approx)
        
        # 设置绘制参数
        shape_name = "未知"
        color = (0, 255, 0)  # 绿色
        
        # 形状分类
        if vertices == 3:
            shape_name = "三角形"
            color = (0, 0, 255)  # 红色
        elif vertices == 4:
            # 判断是矩形还是正方形
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "正方形"
                color = (255, 0, 0)  # 蓝色
            else:
                shape_name = "矩形"
                color = (255, 255, 0)  # 青色
        elif vertices == 5:
            shape_name = "五边形"
            color = (255, 0, 255)  # 紫色
        elif vertices == 6:
            shape_name = "六边形"
            color = (0, 255, 255)  # 黄色
        elif vertices > 6 and complexity < 1.2:
            shape_name = "圆形"
            color = (128, 0, 128)  # 紫红色
        
        # 在图像上绘制轮廓
        cv2.drawContours(result, [contour], -1, color, 2)
        
        # 在图像上标记形状名称
        cv2.putText(result, shape_name, (cx - 20, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 打印形状信息
        print(f"{shape_name}: 面积={area:.2f}, 周长={perimeter:.2f}, 顶点数={vertices}")
    
    return result

# 测试函数
image_path = 'shapes.jpg'  # 替换为实际的图像路径
result_image = detect_shapes(image_path)

if result_image is not None:
    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('识别结果')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

</details>

---

## 练习2：特征点匹配与物体识别

**任务描述：**

+ 创建一个简单的物体识别程序，能够在场景图像中找到目标物体
+ 使用 ORB 算法提取并匹配特征点
+ 在场景图像中标记出目标物体的位置

**提示：**

+ 从目标物体和场景图像中提取 ORB 特征点
+ 使用 BFMatcher 匹配特征点，并应用比值测试筛选匹配
+ 计算单应性矩阵，在场景图像中绘制目标边框

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

def object_detection_with_orb(object_path, scene_path):
    # 读取图像
    obj_image = cv2.imread(object_path)
    scene_image = cv2.imread(scene_path)
    
    if obj_image is None or scene_image is None:
        print("无法读取图像，请检查图像路径")
        return None
    
    # 转换为灰度图像
    obj_gray = cv2.cvtColor(obj_image, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    
    # 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=2000)
    
    # 检测关键点并计算描述符
    kp_obj, des_obj = orb.detectAndCompute(obj_gray, None)
    kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)
    
    print(f"目标图像: 检测到 {len(kp_obj)} 个特征点")
    print(f"场景图像: 检测到 {len(kp_scene)} 个特征点")
    
    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 使用KNN匹配
    matches = bf.knnMatch(des_obj, des_scene, k=2)
    
    # 应用比值测试筛选匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"有效匹配数量: {len(good_matches)}")
    
    # 如果匹配点足够多，计算单应性矩阵并绘制目标边框
    if len(good_matches) >= 10:
        # 从匹配中提取对应点的坐标
        obj_pts = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        scene_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        H, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 5.0)
        
        # 获取目标图像的角点坐标
        h, w = obj_gray.shape
        obj_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # 使用单应性矩阵转换角点坐标到场景图像中
        scene_corners = cv2.perspectiveTransform(obj_corners, H)
        
        # 将场景图像转换为彩色（如果是灰度图像）
        scene_color = scene_image.copy()
        
        # 绘制目标边框
        cv2.polylines(scene_color, [np.int32(scene_corners)], True, (0, 255, 0), 3)
        
        # 绘制显示匹配的特征点
        match_display_image = cv2.drawMatches(
            obj_image, kp_obj, scene_color, kp_scene, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return match_display_image, scene_color
    else:
        print("匹配点不足，无法识别目标")
        return None, None

# 测试函数
object_path = 'object.jpg'  # 替换为实际的目标图像路径
scene_path = 'scene.jpg'    # 替换为实际的场景图像路径

match_image, result_image = object_detection_with_orb(object_path, scene_path)

if match_image is not None and result_image is not None:
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
    plt.title('特征匹配结果')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('目标识别结果')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

</details>

---

## 练习3：结合轮廓和特征点的对象识别

**任务描述：**

+ 开发一个结合轮廓检测和特征匹配的对象识别系统
+ 先使用轮廓检测进行初步目标定位
+ 再使用特征点匹配进行精确识别
+ 比较纯轮廓方法和纯特征点方法的优缺点

**提示：**

+ 使用轮廓检测找到可能的目标区域
+ 在这些区域内进行特征点提取和匹配
+ 对比分析不同方法的识别准确率和效率

<details>
<summary>点击查看答案</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def combined_object_recognition(object_path, scene_path):
    # 读取图像
    obj_image = cv2.imread(object_path)
    scene_image = cv2.imread(scene_path)
    
    if obj_image is None or scene_image is None:
        print("无法读取图像，请检查图像路径")
        return None
    
    # 步骤1: 轮廓检测预处理
    # 转换为灰度图像
    obj_gray = cv2.cvtColor(obj_image, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊处理
    obj_blur = cv2.GaussianBlur(obj_gray, (5, 5), 0)
    scene_blur = cv2.GaussianBlur(scene_gray, (5, 5), 0)
    
    # 边缘检测
    obj_edges = cv2.Canny(obj_blur, 50, 150)
    scene_edges = cv2.Canny(scene_blur, 50, 150)
    
    # 膨胀边缘
    obj_dilated = cv2.dilate(obj_edges, None, iterations=1)
    scene_dilated = cv2.dilate(scene_edges, None, iterations=1)
    
    # 步骤2: 轮廓检测
    obj_contours, _ = cv2.findContours(obj_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scene_contours, _ = cv2.findContours(scene_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"目标图像: 检测到 {len(obj_contours)} 个轮廓")
    print(f"场景图像: 检测到 {len(scene_contours)} 个轮廓")
    
    # 创建结果图像
    result_image = scene_image.copy()
    contours_image = scene_image.copy()
    
    # 绘制场景中的所有轮廓
    cv2.drawContours(contours_image, scene_contours, -1, (255, 0, 0), 2)
    
    # 步骤3: 根据轮廓提取感兴趣区域 (ROI)
    roi_list = []
    for contour in scene_contours:
        area = cv2.contourArea(contour)
        if area < 100:  # 忽略过小的轮廓
            continue
        
        # 获取轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 扩大ROI区域，确保包含完整的物体
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(scene_image.shape[1], x + w + padding)
        y_end = min(scene_image.shape[0], y + h + padding)
        
        # 提取ROI
        roi = scene_image[y_start:y_end, x_start:x_end]
        
        # 保存ROI及其位置信息
        roi_info = {
            'roi': roi,
            'position': (x_start, y_start, x_end, y_end)
        }
        roi_list.append(roi_info)
    
    # 步骤4: 在每个ROI上进行特征点匹配
    # 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=1000)
    
    # 检测目标图像的特征点
    kp_obj, des_obj = orb.detectAndCompute(obj_gray, None)
    print(f"目标图像: 检测到 {len(kp_obj)} 个特征点")
    
    # 创建特征匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 追踪最佳匹配的ROI
    best_match = None
    best_match_count = 0
    best_H = None
    best_roi_index = -1
    
    # 处理每个ROI
    for idx, roi_info in enumerate(roi_list):
        roi = roi_info['roi']
        
        # 如果ROI太小，跳过
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            continue
        
        # 转换ROI为灰度图像
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 检测ROI的特征点
        kp_roi, des_roi = orb.detectAndCompute(roi_gray, None)
        
        if des_roi is None or len(des_roi) < 2:
            continue
        
        print(f"ROI {idx}: 检测到 {len(kp_roi)} 个特征点")
        
        # 特征匹配
        try:
            matches = bf.knnMatch(des_obj, des_roi, k=2)
        except cv2.error:
            # 如果匹配失败（通常是因为特征点太少），跳过
            continue
        
        # 应用比值测试
        good_matches = []
        for match_group in matches:
            if len(match_group) >= 2:
                m, n = match_group
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # 如果有足够多的匹配点
        if len(good_matches) > best_match_count:
            best_match_count = len(good_matches)
            best_match = good_matches
            best_roi_index = idx
            
            # 计算单应性矩阵
            if len(good_matches) >= 10:
                src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                best_H = H
    
    # 步骤5: 标记最佳匹配的物体
    if best_match_count >= 10 and best_roi_index >= 0:
        print(f"发现目标物体在ROI {best_roi_index}，有 {best_match_count} 个匹配点")
        
        # 获取最佳ROI的位置信息
        x_start, y_start, x_end, y_end = roi_list[best_roi_index]['position']
        
        # 在整个场景图像上绘制ROI边界
        cv2.rectangle(result_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)
        
        # 如果有单应性矩阵，绘制精确的物体位置
        if best_H is not None:
            # 获取目标图像的角点坐标
            h, w = obj_gray.shape
            obj_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # 调整角点坐标以适应ROI的位置
            roi_corners = cv2.perspectiveTransform(obj_corners, best_H)
            scene_corners = roi_corners + np.float32([x_start, y_start])
            
            # 绘制物体边框
            cv2.polylines(result_image, [np.int32(scene_corners)], True, (0, 255, 0), 3)
            
        # 准备匹配显示图像
        best_roi = roi_list[best_roi_index]['roi']
        kp_best_roi, _ = orb.detectAndCompute(cv2.cvtColor(best_roi, cv2.COLOR_BGR2GRAY), None)
        
        # 创建一个扩展的结果图像，显示目标图像、所有轮廓和最终结果
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 显示目标图像及其特征点
        obj_with_kp = cv2.drawKeypoints(
            obj_image, kp_obj, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        axes[0, 0].imshow(cv2.cvtColor(obj_with_kp, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('目标图像及特征点')
        axes[0, 0].axis('off')
        
        # 显示场景图像中的轮廓
        axes[0, 1].imshow(cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('场景轮廓检测')
        axes[0, 1].axis('off')
        
        # 显示最佳匹配的ROI
        roi_with_rect = best_roi.copy()
        cv2.rectangle(roi_with_rect, (0, 0), (roi_with_rect.shape[1]-1, roi_with_rect.shape[0]-1), (0, 255, 255), 2)
        axes[1, 0].imshow(cv2.cvtColor(roi_with_rect, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'最佳匹配ROI ({best_match_count}个匹配点)')
        axes[1, 0].axis('off')
        
        # 显示最终结果
        axes[1, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('最终识别结果')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return result_image
    else:
        print("未找到匹配的物体")
        return contours_image

# 测试函数
object_path = 'object.jpg'  # 替换为实际的目标图像路径
scene_path = 'scene.jpg'    # 替换为实际的场景图像路径

result = combined_object_recognition(object_path, scene_path)

if result is not None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('识别结果')
    plt.axis('off')
    plt.show()
```

</details>
