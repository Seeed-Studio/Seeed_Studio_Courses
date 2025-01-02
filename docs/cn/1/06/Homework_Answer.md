# 课后拓展参考答案

## 练习1：系统日志分析器

**任务描述：**

+ 编写一个程序，用于分析边缘设备的系统日志文件，实现以下功能：
  + 读取并解析日志文件内容
  + 统计不同类型错误的出现频率
  + 识别严重的错误模式
  + 生成分析报告
+ 要求能够处理大型日志文件，注意内存使用效率
+ 正确处理文件操作和数据处理过程中可能出现的异常

**提示：**

+ 使用 `with` 语句安全地打开和处理文件
+ 逐行读取文件内容，避免一次性加载大文件
+ 使用字典存储错误统计信息
+ 通过异常处理机制处理文件操作错误

<details>
<summary>点击查看答案</summary>

```python
import re
from datetime import datetime

def analyze_log_file(log_path):
    """分析日志文件，统计错误信息"""
    error_counts = {}
    total_lines = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as file:
            for line in file:
                total_lines += 1
                
# 查找错误信息
                if 'ERROR' in line:
# 使用正则表达式提取错误类型
                    match = re.search(r'ERROR: (\w+)', line)
                    if match:
                        error_type = match.group(1)
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        
# 生成报告
        print(f"日志分析报告 - {datetime.now()}")
        print(f"总行数: {total_lines}")
        print("\n错误统计:")
        for error_type, count in error_counts.items():
            print(f"- {error_type}: {count}次")
            
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {log_path}")
    except Exception as e:
        print(f"分析过程中出错: {e}")

if __name__ == "__main__":
    analyze_log_file("system.log")
```

</details>

---

## 练习2：模型性能记录器

**任务描述：**

+ 创建一个工具记录 AI 模型在边缘设备上的性能数据：
  + 记录每次推理的执行时间
  + 记录推理结果的准确率
  + 记录资源使用情况（内存、CPU使用率等）
  + 定期生成性能报告
+ 实现数据的持久化存储
+ 能够处理记录过程中的异常情况

**提示：**

+ 使用 CSV 格式存储性能数据
+ 定期备份性能记录文件
+ 实现数据记录的异常处理和恢复机制
+ 使用合适的数据结构组织性能指标

<details>
<summary>点击查看答案</summary>

```python
import time
import csv
import os
from datetime import datetime

class PerformanceRecorder:
    def __init__(self, output_dir='performance_logs'):
        self.output_dir = output_dir
        self.log_file = None
        self.writer = None
        self._init_log_file()
        
    def _init_log_file(self):
        """初始化日志文件"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(
                f"{self.output_dir}/perf_log_{timestamp}.csv",
                'w',
                newline=''
            )
            self.writer = csv.writer(self.log_file)
# 写入CSV头部
            self.writer.writerow([
                'timestamp', 'model_name', 'inference_time_ms',
                'accuracy', 'memory_usage_mb', 'cpu_usage_percent'
            ])
        except Exception as e:
            print(f"初始化日志文件失败: {e}")
            raise
            
    def record_inference(self, model_name, inference_time, accuracy, 
                        memory_usage, cpu_usage):
        """记录一次推理的性能数据"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.writer.writerow([
                timestamp,
                model_name,
                f"{inference_time:.2f}",
                f"{accuracy:.4f}",
                f"{memory_usage:.1f}",
                f"{cpu_usage:.1f}"
            ])
            self.log_file.flush()  # 确保数据被写入磁盘
            
        except Exception as e:
            print(f"记录性能数据失败: {e}")
# 尝试重新初始化日志文件
            self._init_log_file()
            
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()

if __name__ == "__main__":
# 测试性能记录器
    recorder = PerformanceRecorder()
    try:
# 模拟记录多次推理性能
        for i in range(3):
            recorder.record_inference(
                model_name="yolov5s",
                inference_time=random.uniform(20, 50),  # 20-50ms
                accuracy=random.uniform(0.8, 0.95),     # 80-95%
                memory_usage=random.uniform(200, 500),  # 200-500MB
                cpu_usage=random.uniform(20, 80)        # 20-80%
            )
            time.sleep(1)
    finally:
        recorder.close()
```

</details>

---

## 练习3：配置文件比较工具

**任务描述：**

+ 开发一个工具用于比较两个 AI 模型配置文件的差异：
  + 支持比较不同版本的配置文件
  + 识别新增、删除和修改的配置项
  + 生成详细的差异报告
  + 提供配置迁移建议
+ 支持 JSON 格式的配置文件
+ 正确处理配置文件的格式错误和访问错误

**提示：**

+ 使用 JSON 模块处理配置文件
+ 实现深度比较以处理嵌套的配置项
+ 生成清晰的差异报告
+ 处理配置文件读取和解析时的异常

<details>
<summary>点击查看答案</summary>

```python
import json
from typing import Dict, Any, Tuple

class ConfigComparator:
    def __init__(self, old_path: str, new_path: str):
        self.old_path = old_path
        self.new_path = new_path
        self.old_config = None
        self.new_config = None
        
    def load_configs(self):
        """加载两个配置文件"""
        try:
            with open(self.old_path, 'r') as f1, open(self.new_path, 'r') as f2:
                self.old_config = json.load(f1)
                self.new_config = json.load(f2)
        except FileNotFoundError as e:
            print(f"找不到配置文件: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            raise
            
    def compare_configs(self) -> Dict[str, Tuple]:
        """比较两个配置文件的差异"""
        if not (self.old_config and self.new_config):
            self.load_configs()
            
        differences = {}
        
# 查找修改和删除的配置项
        for key in self.old_config:
            if key not in self.new_config:
                differences[key] = ('removed', self.old_config[key], None)
            elif self.old_config[key] != self.new_config[key]:
                differences[key] = (
                    'modified',
                    self.old_config[key],
                    self.new_config[key]
                )
                
# 查找新增的配置项
        for key in self.new_config:
            if key not in self.old_config:
                differences[key] = ('added', None, self.new_config[key])
                
        return differences
        
    def generate_report(self):
        """生成差异报告"""
        try:
            differences = self.compare_configs()
            
            print("\n=== 配置文件差异报告 ===")
            print(f"对比文件:")
            print(f"- 旧配置: {self.old_path}")
            print(f"- 新配置: {self.new_path}\n")
            
            if not differences:
                print("没有发现差异")
                return
                
# 按变更类型分类显示
            for change_type in ['added', 'removed', 'modified']:
                changes = {k: v for k, v in differences.items() 
                        if v[0] == change_type}
                if changes:
                    print(f"\n{change_type.title()} 的配置项:")
                    for key, (_, old_val, new_val) in changes.items():
                        if change_type == 'modified':
                            print(f"- {key}:")
                            print(f"  旧值: {old_val}")
                            print(f"  新值: {new_val}")
                        else:
                            print(f"- {key}: {new_val or old_val}")
                            
# 提供迁移建议
            self._provide_migration_advice(differences)
            
        except Exception as e:
            print(f"生成报告时出错: {e}")
            
    def _provide_migration_advice(self, differences):
        """生成配置迁移建议"""
        print("\n迁移建议:")
        for key, (change_type, old_val, new_val) in differences.items():
            if change_type == 'added':
                print(f"- 需要添加新配置项: {key}")
            elif change_type == 'removed':
                print(f"- 注意: 配置项 {key} 已被移除")
            elif change_type == 'modified':
                print(f"- 需要更新配置项: {key}")

if __name__ == "__main__":
# 测试配置比较工具
    comparator = ConfigComparator('old_config.json', 'new_config.json')
    comparator.generate_report()
```

</details>
