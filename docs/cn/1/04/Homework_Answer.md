# 课后拓展参考答案

## 练习1：AI模型置信度评估工具

**任务描述：**

+ 编写一个程序，包含函数用于评估AI模型的预测置信度（范围0到1的浮点数）。
+ 创建函数生成5个模拟的置信度值。
+ 创建函数进行置信度评估：
  + 如果置信度大于0.8，输出“高可信度”。
  + 如果置信度在0.5到0.8之间，输出“中等可信度”。
  + 如果置信度小于0.5，输出“低可信度”。

**提示：**

+ 使用`random.uniform(0, 1)`生成0到1之间的随机浮点数。
+ 使用`round(number, 2)`保留两位小数。
+ 将相关函数放在同一个模块中。

<details>
<summary>点击查看答案</summary>

```python
import random

def generate_confidence():
    """生成模拟的置信度值"""
    return round(random.uniform(0, 1), 2)

def evaluate_confidence(score):
    """评估置信度"""
    if score > 0.8:
        return "高可信度"
    elif score >= 0.5:
        return "中等可信度"
    else:
        return "低可信度"

# 主程序
for i in range(1, 6):
    confidence = generate_confidence()
    result = evaluate_confidence(confidence)
    print(f"预测 {i}: 置信度 {confidence}, {result}")
```

</details>

---

## 练习2：传感器数据处理模块

**任务描述：**

+ 创建一个模块，包含以下函数：
  + 生成模拟的传感器数据（范围0到100的整数）。
  + 计算数据的平均值。
  + 查找最大值和最小值。
+ 在主程序中使用这些函数处理10个数据点。
+ 生成数据报告，显示统计结果。

**提示：**

+ 将所有函数放在一个独立的模块文件中。
+ 使用列表存储数据。
+ 使用函数参数传递数据。

<details>
<summary>点击查看答案</summary>

```python
# sensor_data.py
import random

def generate_data(count):
    """生成模拟传感器数据"""
    data = []
    for _ in range(count):
        data.append(random.randint(0, 100))
    return data

def calculate_average(data):
    """计算平均值"""
    return sum(data) / len(data)

def find_max_min(data):
    """查找最大值和最小值"""
    return max(data), min(data)

# main.py
import sensor_data

# 生成并处理数据
readings = sensor_data.generate_data(10)
avg = sensor_data.calculate_average(readings)
max_val, min_val = sensor_data.find_max_min(readings)

# 显示报告
print(f"传感器数据: {readings}")
print(f"平均值: {avg:.2f}")
print(f"最大值: {max_val}")
print(f"最小值: {min_val}")
```

</details>

---

## 练习3：模型预测时间统计工具

**任务描述：**

+ 创建函数模拟AI模型的预测过程。
+ 统计每次预测的执行时间。
+ 创建函数计算并显示：
  + 平均执行时间。
  + 最长执行时间。
  + 最短执行时间。
+ 进行5次测试并生成报告。

**提示：**

+ 使用`time`模块的`time()`函数记录时间。
+ 使用`sleep()`函数模拟处理时间。
+ 将时间统计相关的函数放在独立模块中。

<details>
<summary>点击查看答案</summary>

```python
# time_stats.py
import time
import random

def simulate_prediction():
    """模拟预测过程"""
    start_time = time.time()
    # 模拟处理时间
    time.sleep(random.uniform(0.1, 0.5))
    end_time = time.time()
    return end_time - start_time

def process_time_stats(times):
    """处理时间统计"""
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    return avg_time, max_time, min_time

# 主程序
prediction_times = []
for i in range(5):
    exec_time = simulate_prediction()
    prediction_times.append(exec_time)
    print(f"预测 {i+1}: {exec_time:.3f} 秒")

avg, max_t, min_t = process_time_stats(prediction_times)
print(f"\n统计结果:")
print(f"平均时间: {avg:.3f} 秒")
print(f"最长时间: {max_t:.3f} 秒")
print(f"最短时间: {min_t:.3f} 秒")
```

</details>
