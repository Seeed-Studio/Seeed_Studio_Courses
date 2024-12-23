# 课后拓展参考答案

## 练习1：多传感器数据采集系统设计

**任务描述：**

+ 创建一个系统来管理多个传感器的数据采集。
+ 包含温度、湿度和光照三种类型的传感器。
+ 使用字典存储传感器配置和阈值信息。
+ 使用列表记录每个传感器的历史数据，最多保存100条记录。
+ 使用集合记录传感器异常状态，当数据超过阈值时，记录异常。
+ 提供功能来获取每个传感器的统计信息（当前值、平均值、最大值、最小值）。
+ 显示所有异常记录。

**提示：**

+ 使用字典嵌套存储每个传感器的配置。
+ 使用列表切片保持每个传感器的历史记录（最多100条）。
+ 使用集合自动去除重复的异常记录。

<details>
<summary>点击查看答案</summary>

```python
import random
from datetime import datetime

# 传感器配置
sensor_config = {
    'temperature': {'threshold': 30, 'unit': '°C'},
    'humidity': {'threshold': 80, 'unit': '%'},
    'light': {'threshold': 1000, 'unit': 'lux'}
}

# 历史数据(每个传感器最多保存100条记录)
sensor_history = {
    'temperature': [],
    'humidity': [],
    'light': []
}

# 异常记录
alerts = set()

def collect_data():
    """采集所有传感器数据"""
    data = {
        'temperature': round(random.uniform(20, 35), 2),
        'humidity': round(random.uniform(40, 90), 2),
        'light': round(random.uniform(500, 1500), 2),
        'timestamp': datetime.now()
    }
    
    # 存储和检查数据
    for sensor_type, value in data.items():
        if sensor_type == 'timestamp':
            continue
            
        # 保存数据
        sensor_history[sensor_type].append({
            'value': value,
            'timestamp': data['timestamp']
        })
        
        # 保持最近100条记录
        if len(sensor_history[sensor_type]) > 100:
            sensor_history[sensor_type].pop(0)
        
        # 检查异常
        if value > sensor_config[sensor_type]['threshold']:
            alert = f"{sensor_type}_HIGH_{data['timestamp'].strftime('%H:%M:%S')}"
            alerts.add(alert)
    
    return data

def get_sensor_stats(sensor_type):
    """获取指定传感器的统计信息"""
    if not sensor_history[sensor_type]:
        return None
        
    values = [record['value'] for record in sensor_history[sensor_type]]
    return {
        'current': values[-1],
        'average': sum(values) / len(values),
        'maximum': max(values),
        'minimum': min(values),
        'records': len(values),
        'unit': sensor_config[sensor_type]['unit']
    }

def get_alerts():
    """获取所有异常记录"""
    return sorted(list(alerts))

# 测试代码
if __name__ == "__main__":
    print("多传感器数据采集测试:\n")
    
    # 模拟采集5次数据
    for i in range(5):
        data = collect_data()
        print(f"第 {i+1} 次采集结果:")
        for sensor_type, value in data.items():
            if sensor_type != 'timestamp':
                print(f"{sensor_type}: {value}{sensor_config[sensor_type]['unit']}")
        print()
    
    # 显示统计信息
    print("传感器统计信息:")
    for sensor_type in sensor_config.keys():
        stats = get_sensor_stats(sensor_type)
        if stats:
            print(f"\n{sensor_type}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    # 显示异常记录
    print("\n异常记录:")
    for alert in get_alerts():
        print(f"  {alert}")
```

</details>

---

## 练习2：目标检测结果分析器设计

**任务描述：**

+ 设计一个目标检测结果分析器。
+ 统计不同类别目标的出现次数。
+ 计算每个类别的平均置信度。
+ 找出最频繁出现的目标类别。
+ 提供接口来查看统计信息和类别详情。

**提示：**

+ 使用字典记录每个类别的计数和累计置信度。
+ 使用列表存储原始检测结果以便回溯。
+ 合理组织数据结构以提高查询效率。

<details>
<summary>点击查看答案</summary>

```python
# 全局统计数据
class_counts = {}  # 类别计数
class_confidence = {}  # 类别置信度累计
detection_history = []  # 原始检测记录

def update_statistics(detections):
    """更新检测统计信息
    
    Args:
        detections: list of tuple (class_name, confidence)
    """
    for class_name, confidence in detections:
        # 更新类别计数
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidence[class_name] = 0.0
        
        class_counts[class_name] += 1
        class_confidence[class_name] += confidence
    
    # 保存原始记录
    detection_history.append(detections)

def get_average_confidence(class_name):
    """获取指定类别的平均置信度"""
    if class_name in class_counts and class_counts[class_name] > 0:
        return class_confidence[class_name] / class_counts[class_name]
    return 0.0

def get_most_frequent_class():
    """获取出现最频繁的类别"""
    if not class_counts:
        return None
    return max(class_counts.items(), key=lambda x: x[1])

def get_statistics():
    """获取统计报告"""
    stats = {
        'total_detections': sum(class_counts.values()),
        'class_counts': class_counts.copy(),
        'average_confidences': {},
        'most_frequent_class': get_most_frequent_class()
    }
    
    # 计算每个类别的平均置信度
    for class_name in class_counts:
        stats['average_confidences'][class_name] = round(
            get_average_confidence(class_name), 3)
    
    return stats

# 测试代码
if __name__ == "__main__":
    # 模拟一些检测结果
    test_detections = [
        [('person', 0.95), ('car', 0.87)],
        [('car', 0.92), ('person', 0.85)],
        [('bike', 0.78), ('person', 0.90)]
    ]
    
    print("目标检测结果分析测试:\n")
    
    # 处理检测结果
    for i, detections in enumerate(test_detections, 1):
        print(f"处理第 {i} 帧检测结果:")
        print(f"检测到的目标: {detections}")
        update_statistics(detections)
        print()
    
    # 获取统计信息
    stats = get_statistics()
    print("统计结果:")
    for key, value in stats.items():
        print(f"\n{key}:")
        print(f"  {value}")
```

</details>

---

## 练习3：边缘设备状态监控程序设计

**任务描述：**

+ 创建一个边缘设备状态监控程序。
+ 使用不同的数据结构跟踪设备的 CPU 使用率、内存使用率和温度。
+ 当任何指标超过预设阈值时，将其记录到告警历史中。
+ 能够显示实时的状态并提供历史统计信息。

**提示：**

+ 使用元组存储阈值配置（正常、警告、危险）。
+ 使用字典存储当前状态和配置信息。
+ 使用列表记录历史数据。
+ 使用集合存储已触发的告警类型。

<details>
<summary>点击查看答案</summary>

```python
import random
from datetime import datetime

# 阈值配置 (正常阈值, 警告阈值, 危险阈值)
THRESHOLDS = {
    'cpu': (70, 85, 95),
    'memory': (60, 80, 90),
    'temperature': (40, 60, 75)
}

# 历史数据
history = {
    'cpu': [],
    'memory': [],
    'temperature': []
}

# 告警记录
alerts = set()

def collect_metrics():
    """采集当前设备指标"""
    return {
        'cpu': round(random.uniform(30, 100), 1),
        'memory': round(random.uniform(40, 100), 1),
        'temperature': round(random.uniform(30, 80), 1),
        'timestamp': datetime.now()
    }

def check_threshold(metric_type, value):
    """检查指标是否超过阈值
    
    Returns:
        str: 状态级别 ('normal', 'warning', 'critical')
    """
    normal, warning, danger = THRESHOLDS[metric_type]
    
    if value > danger:
        alerts.add(f"{metric_type.upper()}_CRITICAL_{datetime.now().strftime('%H:%M:%S')}")
        return 'critical'
    elif value > warning:
        alerts.add(f"{metric_type.upper()}_WARNING_{datetime.now().strftime('%H:%M:%S')}")
        return 'warning'
    return 'normal'

def update_history(metrics):
    """更新历史记录"""
    for metric_type, value in metrics.items():
        if metric_type != 'timestamp':
            history[metric_type].append({
                'value': value,
                'timestamp': metrics['timestamp']
            })
            # 保持最近100条记录
            if len(history[metric_type]) > 100:
                history[metric_type].pop(0)

def get_status_level(metrics):
    """获取当前整体状态级别"""
    status_levels = []
    for metric_type, value in metrics.items():
        if metric_type != 'timestamp':
            status_levels.append(check_threshold(metric_type, value))
    
    return 'critical' if 'critical' in status_levels else \
        'warning' if 'warning' in status_levels else \
        'normal'

def get_statistics():
    """获取统计信息"""
    stats = {}
    
    for metric_type in THRESHOLDS.keys():
        if not history[metric_type]:
            continue
            
        values = [record['value'] for record in history[metric_type]]
        stats[metric_type] = {
            'current': values[-1],
            'average': sum(values) / len(values),
            'maximum': max(values),
            'minimum': min(values)
        }
    
    stats['alerts'] = sorted(list(alerts))
    return stats

# 测试代码
if __name__ == "__main__":
    print("设备状态监控测试:\n")
    
    # 模拟监控5个时间点
    for i in range(5):
        # 采集数据
        metrics = collect_metrics()
        
        # 更新历史记录
        update_history(metrics)
        
        # 获取当前状态
        status = get_status_level(metrics)
        
        print(f"时间点 {i+1}:")
        for metric_type, value in metrics.items():
            if metric_type != 'timestamp':
                print(f"{metric_type}: {value}")
        print(f"整体状态: {status}\n")
    
    # 显示统计信息
    print("统计信息:")
    stats = get_statistics()
    for metric_type, metric_stats in stats.items():
        print(f"\n{metric_type}:")
        if metric_type != 'alerts':
            for key, value in metric_stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {metric_stats}")
```

</details>
