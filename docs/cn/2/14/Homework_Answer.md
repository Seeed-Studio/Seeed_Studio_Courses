# 课后拓展参考答案

## 练习1：实时视频流处理

**任务描述**：

+ 修改当前项目，使其能够处理摄像头的实时视频流而非视频文件
+ 添加帧率计算和显示功能，实时显示处理性能
+ 优化处理流程，确保应用能够流畅运行

**提示**：

+ 将 `VideoReader` 类改为支持摄像头输入
+ 使用 `time` 模块来计算和显示帧率
+ 考虑降低处理分辨率或简化处理步骤以提高性能

<details>
<summary>点击查看答案</summary>

```python
# 修改 VideoReader 类，支持摄像头输入
def open_camera(self, camera_id=0):
    """
    打开摄像头
    
    Args:
        camera_id: 摄像头 ID，默认为 0（通常是内置摄像头）
        
    Returns:
        bool: 是否成功打开摄像头
    """
    # 释放之前可能打开的视频
    self.release()

    # 打开摄像头
    self.video_path = f"Camera-{camera_id}"
    self.cap = cv2.VideoCapture(camera_id)
    self.is_opened = self.cap.isOpened()

    if self.is_opened:
        # 获取视频属性
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_index = 0
        self.is_playing = True

        print(f"摄像头已连接: {self.video_path}")
        print(f"分辨率: {self.frame_width}x{self.frame_height}")
        return True
    else:
        print(f"无法连接摄像头")
        return False

# 在 VideoProcessor 类中添加帧率计算
def run_camera(self, camera_id=0):
    """
    运行摄像头处理主循环
    
    Args:
        camera_id: 摄像头 ID
    """
    if not self.video_reader.open_camera(camera_id):
        return

    # 显示帮助信息
    self.ui_helper.display_help()

    # 初始化帧率计算
    frame_counter = 0
    fps = 0
    prev_time = time.time()

    while True:
        # 读取视频帧
        ret, frame = self.video_reader.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 处理帧
        processed_frame = self.process_frame(frame)

        # 计算和显示帧率
        frame_counter += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = frame_counter
            frame_counter = 0
            prev_time = current_time

        cv2.putText(processed_frame, f"FPS: {fps}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 录制视频
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(processed_frame)

        # 显示处理后的帧
        cv2.imshow('综合视频处理器', processed_frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if not self.handle_input(key):
            break

    # 释放资源
    self.video_reader.release()
    if self.video_writer is not None:
        self.video_writer.release()
    cv2.destroyAllWindows()
    print("程序已退出")
```

</details>

---

## 练习2：基本形状分析与测量

**任务描述**：

+ 在当前项目基础上，添加一个形状分析模式，能够分析视频中的基本几何形状
+ 测量并显示检测到的形状的面积、周长、近似度等基本特征
+ 支持识别和标记基本形状（圆形、三角形、矩形、正方形等）

**提示**：

+ 利用已学的轮廓检测功能作为基础
+ 使用 `cv2.approxPolyDP()` 函数近似轮廓形状
+ 结合第 12 课学习的轮廓特征计算方法对形状进行分析
+ 计算一些基本的形状特征，如圆形度 (4π×面积/周长²)，帮助识别圆形

<details>
<summary>点击查看答案</summary>

```python
# 在 FeatureDetector 类中扩展形状分析功能
def analyze_shapes(self, frame):
    """
    形状分析
    
    Args:
        frame: 输入图像
        
    Returns:
        标记形状分析后的图像
    """
    # 创建结果图像的副本
    result = frame.copy()
    
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 阈值处理
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 分析每个轮廓
    shape_count = 0
    for contour in contours:
        # 计算面积，忽略小轮廓
        area = cv2.contourArea(contour)
        if area < 500:  # 忽略小区域
            continue
            
        shape_count += 1
        
        # 计算周长
        perimeter = cv2.arcLength(contour, True)
        
        # 轮廓近似
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 识别形状
        shape_name = "未知"
        vertices = len(approx)
        
        if vertices == 3:
            shape_name = "三角形"
        elif vertices == 4:
            # 计算长宽比，判断是矩形还是正方形
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            shape_name = "正方形" if 0.95 <= aspect_ratio <= 1.05 else "矩形"
        elif vertices == 5:
            shape_name = "五边形"
        elif vertices == 6:
            shape_name = "六边形"
        elif vertices > 8:
            # 计算圆形度
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.8:
                shape_name = "圆形"
        
        # 获取轮廓的极值点
        x, y, w, h = cv2.boundingRect(contour)
        
        # 绘制轮廓
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)
        
        # 标记形状名称
        cv2.putText(result, shape_name, (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 显示测量信息
        info_text = f"面积: {int(area)}, 周长: {int(perimeter)}"
        cv2.putText(result, info_text, (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 显示检测到的形状数量
    cv2.putText(result, f"检测到 {shape_count} 个形状", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result

# 在 VideoProcessor 类的 __init__ 方法中添加新模式常量
MODE_SHAPES = 8  # 形状分析

# 在 __init__ 方法中更新模式名称映射
self.mode_names[self.MODE_SHAPES] = "形状分析"

# 在 process_frame 方法中添加对应处理
elif self.current_mode == self.MODE_SHAPES:
    processed_frame = self.feature_detector.analyze_shapes(frame)
```

</details>
