# 课后拓展参考答案

## 练习1：调整参数优化人脸检测

**任务描述**：

+ 通过调整`detectMultiScale`方法的参数，优化人脸检测效果
+ 系统地测试不同参数组合对检测结果的影响
+ 找出特定场景下的最佳参数配置

**提示**：

+ 创建一个测试框架，可以方便地调整和记录不同参数
+ 使用多种测试图像，包括不同光照条件、多人场景、不同距离的人脸
+ 记录每组参数下的检测结果、漏检率和误检率

<details>
<summary>点击查看答案</summary>

```python
import cv2

# 加载 Haar 级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
img = cv2.imread('face.jpg')
if img is None:
    print('无法读取图像')
    exit()

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 调整参数进行人脸检测
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,   # 可调整的参数
    minNeighbors=3,    # 可调整的参数
    minSize=(50, 50),  # 可调整的参数
    maxSize=(400, 400) # 可调整的参数
)

# 绘制检测结果
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow('Optimized Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

</details>

---

## 练习2：多特征联合检测

**任务描述**：

+ 开发一个能同时检测人脸、眼睛和微笑的系统
+ 实现简单的表情识别功能（如判断是否微笑）
+ 在实时视频中展示检测结果

**提示**：

+ 使用多个级联分类器模型，如`haarcascade_eye.xml`和`haarcascade_smile.xml`
+ 先检测人脸，然后在人脸区域内检测眼睛和微笑
+ 设计合理的显示方式，如不同颜色的矩形框或文字标签

<details>
<summary>点击查看答案</summary>

```python
import cv2

# 加载 Haar 级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # 绘制人脸矩形框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 提取人脸区域
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # 在人脸区域内检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # 在人脸区域内检测微笑
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

        # 简单表情识别
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Neutral', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    # 显示结果
    cv2.imshow('Multi-feature Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

</details>
