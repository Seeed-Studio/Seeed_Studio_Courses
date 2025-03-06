# 课后拓展参考答案

## 练习1：视频分辨率转换器

**任务描述：**

+ 开发一个视频分辨率转换工具，能够实时调整视频分辨率
+ 支持多种常见分辨率的切换（如 640x480、1280x720）
+ 显示当前分辨率信息

**提示：**：

+ 使用 VideoCapture 设置摄像头分辨率
+ 添加帧率显示功能
+ 实现实时分辨率切换

<details>
<summary>点击查看答案</summary>

```python
import cv2
import time
from datetime import datetime

class ResolutionConverter:
    def __init__(self):
        self.cap = None
        self.current_resolution = 0
        self.resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080)
        ]
        
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.set_resolution(self.current_resolution)
        
    def set_resolution(self, index):
        width, height = self.resolutions[index]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
    def show_info(self, frame):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        cv2.putText(frame, f"Resolution: {width}x{height}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
                    
    def run(self):
        self.initialize_camera()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                self.show_info(frame)
                cv2.imshow('Resolution Converter', frame)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.current_resolution = (self.current_resolution + 1) % len(self.resolutions)
                    self.set_resolution(self.current_resolution)
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    converter = ResolutionConverter()
    converter.run()

if __name__ == '__main__':
    main()
```

</details>

---

## 练习2：视频时间戳记录器

**任务描述：**

+ 开发一个视频录制系统，在视频上添加时间戳和自定义文本
+ 支持开始/停止录制功能
+ 在视频中显示录制状态和时长

**提示：**

+ 使用 VideoWriter 保存视频
+ 添加文本标注显示时间信息
+ 实现录制状态管理

<details>
<summary>点击查看答案</summary>

```python
import cv2
import time
from datetime import datetime

class TimestampRecorder:
"""视频时间戳记录器"""
def __init__(self):
    self.cap = None
    self.writer = None
    self.is_recording = False
    self.start_time = None
    
def initialize_camera(self):
    """初始化摄像头"""
    self.cap = cv2.VideoCapture(0)
    if not self.cap.isOpened():
        raise RuntimeError("无法打开摄像头")
        
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
def create_writer(self):
    """创建视频写入器"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = f"video_{int(time.time())}.avi"
    frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    self.writer = cv2.VideoWriter(
        filename,
        fourcc,
        20.0,
        frame_size
    )
    self.start_time = time.time()
    
def add_timestamp(self, frame):
    """添加时间戳和录制信息"""
    # 添加当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
                
    # 添加录制状态
    status = "Recording" if self.is_recording else "Standby"
    cv2.putText(frame, f"Status: {status}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255) if self.is_recording else (0, 255, 0), 2)
                
    # 添加录制时长
    if self.is_recording:
        duration = int(time.time() - self.start_time)
        cv2.putText(frame, f"Duration: {duration}s",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
                    
def toggle_recording(self):
    """切换录制状态"""
    if not self.is_recording:
        self.create_writer()
        self.is_recording = True
    else:
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        self.start_time = None
        
def run(self):
    """运行录制器"""
    self.initialize_camera()
    
    try:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.add_timestamp(frame)
            
            if self.is_recording:
                self.writer.write(frame)
                
            cv2.imshow('Timestamp Recorder', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.toggle_recording()
                
    finally:
        self.cleanup()
        
def cleanup(self):
    """清理资源"""
    if self.cap:
        self.cap.release()
    if self.writer:
        self.writer.release()
    cv2.destroyAllWindows()

def main():
"""主程序入口"""
recorder = TimestampRecorder()
try:
    recorder.run()
except Exception as e:
    print(f"错误: {e}")

if __name__ == '__main__':
main()
```

</details>
