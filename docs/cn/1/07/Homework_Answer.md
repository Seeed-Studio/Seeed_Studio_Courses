# 课后拓展参考答案

## 练习1：获取真实设备数据

**任务描述：**

+ 修改 `Monitor` 类中的数据采集方法，使用 `psutil` 库获取实际设备数据，包括 CPU 温度、内存使用率和 GPU 使用率。
+ 处理不同操作系统和硬件的兼容性问题，确保在各种环境下都能正确获取数据。
+ 实现错误处理和重试机制，确保在数据采集过程中遇到问题时系统能够稳定运行。

**提示：**

+ 参考现有的 Monitor 类结构
+ 使用 `psutil` 库获取 CPU 和内存数据
+ 对于 GPU 数据，根据实际硬件选择合适的获取方式
+ 确保将获取到的数据传递给 `DeviceStatus` 类进行管理和验证。

<details>
<summary>点击查看答案</summary>

```python
import time
import asyncio
import threading
import psutil
from typing import Optional
from device_monitor.device_status import DeviceStatus
from device_monitor.logger import Logger
from jtop import jtop

class Monitor:
    """
    Monitor 类负责管理监控生命周期，协调各个组件工作，实现优雅的启动和停止。
    它与 DeviceStatus 和 Logger 类集成，支持异步监控、可配置的监控间隔，并提供扩展接口。
    """

    def __init__(self, device_status: 'DeviceStatus', logger: 'Logger', monitor_interval: int = 5):
        """
        初始化 Monitor 类。

        :param device_status: DeviceStatus 实例，用于获取设备状态
        :param logger: Logger 实例，用于记录日志
        :param monitor_interval: 监控间隔（单位：秒）
        """
        self.device_status = device_status
        self.logger = logger
        self.monitor_interval = monitor_interval
        self._stop_flag = threading.Event()
        self._monitor_thread = None
        self.jtop_instance = None

    async def _monitor_device(self):
        """
        异步监控设备状态并记录日志。
        """
        # 初始化 jtop
        try:
            self.jtop_instance = jtop()
            self.jtop_instance.start()
            self.logger.info("jtop 已启动。")
        except Exception as e:
            self.logger.error(f"无法启动 jtop: {e}")
            return

        while not self._stop_flag.is_set():
            try:
                # 更新 jtop 数据
                self.jtop_instance.update()

                # 获取设备数据
                cpu_temp = self.get_cpu_temperature()
                mem_usage = self.get_memory_usage()
                gpu_usage = self.get_gpu_usage()

                # 更新 DeviceStatus 对象
                self.device_status.cpu_temp = cpu_temp
                self.device_status.mem_usage = mem_usage
                self.device_status.gpu_usage = gpu_usage

                # 获取设备健康状态
                health_status = self.device_status.check_health()

                # 记录设备状态
                self.logger.info(f"设备状态: {health_status}")
                self.logger.debug(f"设备详情: {str(self.device_status)}")

                # 按照指定间隔暂停
                await asyncio.sleep(self.monitor_interval)
            except Exception as e:
                # 处理监控过程中出现的任何异常
                self.logger.error(f"监控过程中发生错误: {e}")
                await asyncio.sleep(5)  # 错误恢复等待时间

        # 停止 jtop
        self.jtop_instance.close()
        self.logger.info("jtop 已停止。")

    def _start_monitoring(self):
        """
        启动监控线程，执行异步监控任务。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._monitor_device())

    def start(self):
        """
        启动监控系统，开始监控设备状态。
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self.logger.warning("监控系统已经在运行中！")
            return

        self._stop_flag.clear()
        self._monitor_thread = threading.Thread(target=self._start_monitoring)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        self.logger.info("监控系统已启动。")

    def stop(self):
        """
        停止监控系统，结束监控任务。
        """
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self.logger.warning("监控系统未运行！")
            return

        self._stop_flag.set()
        self._monitor_thread.join()
        self.logger.info("监控系统已停止。")

    def restart(self):
        """
        重启监控系统。
        """
        self.stop()
        time.sleep(2)  # 等待一段时间确保系统已经停止
        self.start()
        self.logger.info("监控系统已重启。")

    def get_status(self) -> str:
        """
        获取当前监控系统的状态。
        :return: 当前监控系统的状态
        """
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            return "监控系统未启动"
        return "监控系统正在运行"

    def set_monitor_interval(self, interval: int):
        """
        设置监控间隔。
        :param interval: 监控间隔（单位：秒）
        """
        if interval <= 0:
            self.logger.warning("监控间隔必须为正数！")
            return
        self.monitor_interval = interval
        self.logger.info(f"监控间隔已设置为 {interval} 秒。")

    def extend_monitoring(self, additional_monitor: Optional['Monitor'] = None):
        """
        提供扩展接口，可以添加其他监控任务。
        :param additional_monitor: 可选的扩展监控实例
        """
        if additional_monitor:
            self.logger.info(f"扩展监控系统，添加额外监控任务: {additional_monitor.get_status()}")
            # 这里可以添加逻辑来协调多个监控任务
            additional_monitor.start()

    def collect_device_data(self) -> tuple:
        """
        收集设备的 CPU 温度、内存使用率和 GPU 使用率。

        :return: (cpu_temp, mem_usage, gpu_usage)
        """
        cpu_temp = self.get_cpu_temperature()
        mem_usage = self.get_memory_usage()
        gpu_usage = self.get_gpu_usage()
        return cpu_temp, mem_usage, gpu_usage

    def get_cpu_temperature(self) -> float:
        """
        获取 CPU 温度。

        :return: CPU 温度（摄氏度）
        """
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                raise ValueError("无法读取 CPU 温度信息。")
            # 假设温度传感器名称为 'cpu-thermal' 或根据实际情况调整
            for name, entries in temps.items():
                for entry in entries:
                    if 'cpu' in entry.label.lower() or 'core' in entry.label.lower():
                        self.logger.debug(f"读取到 CPU 温度: {entry.current}°C")
                        return entry.current
            # 如果没有找到特定标签，返回第一个温度传感器的值
            first_temp = next(iter(temps.values()))[0].current
            self.logger.debug(f"未找到特定 CPU 温度标签，使用默认温度: {first_temp}°C")
            return first_temp
        except Exception as e:
            self.logger.error(f"获取 CPU 温度失败: {e}")
            # 重试机制或默认值
            return 0.0

    def get_memory_usage(self) -> float:
        """
        获取内存使用率。

        :return: 内存使用率（百分比）
        """
        try:
            mem = psutil.virtual_memory()
            self.logger.debug(f"读取到内存使用率: {mem.percent}%")
            return mem.percent
        except Exception as e:
            self.logger.error(f"获取内存使用率失败: {e}")
            # 重试机制或默认值
            return 0.0

    def get_gpu_usage(self) -> float:
        """
        获取 GPU 使用率。
        使用 jetson-stats 的 jtop 库获取 GPU 使用率。
        """

        try:
            if not self.jtop_instance:
                self.logger.error("jtop 实例未初始化。")
                return 0.0

            # 调试：打印 perf 字段内容
            self.logger.debug(f"jtop perf 数据: {self.jtop_instance.perf}")

            # 假设 'gpu' 字段存在，根据实际 API 调整字段名称
            gpu_usage = self.jtop_instance.perf.get('gpu', 0.0)
            self.logger.debug(f"读取到 GPU 使用率: {gpu_usage}%")
            return float(gpu_usage)
        except KeyError:
            self.logger.error("无法找到 GPU 使用率字段。")
            return 0.0
        except Exception as e:
            self.logger.error(f"获取 GPU 使用率失败: {e}")
            return 0.0
```

</details>

---

## 练习2：设备状态分析

**任务描述：**

+ 实现一个 `DeviceAnalyzer` 类，用于分析设备的运行状态趋势。
+ 该类应能够记录历史数据，分析趋势（如温度上升或下降），并生成简单的分析报告。
+ 将 `DeviceAnalyzer` 类与现有的监控系统集成，使其能够定期接收设备状态数据并进行分析。

**提示：**

+ 使用 `collections.deque` 来存储历史数据，便于维护固定大小的滑动窗口。
+ 设计分析方法，如计算平均值、最大值、最小值，判断趋势等。
+ 结合现有的 `Logger` 类记录分析结果。

<details>
<summary>点击查看答案</summary>

首先，在 device_monitor 目录下创建一个新的文件 device_analyzer.py，并实现 DeviceAnalyzer 类。

```python
from collections import deque
from typing import List, Tuple
from device_monitor.device_status import DeviceStatus
from device_monitor.logger import Logger

class DeviceAnalyzer:
    """
    DeviceAnalyzer 类用于分析设备的运行状态趋势。
    它记录历史数据，分析趋势（如温度上升或下降），并生成简单的分析报告。
    """

    def __init__(self, history_size: int = 60, logger: Logger = None):
        """
        初始化 DeviceAnalyzer 对象。

        :param history_size: 历史数据的最大记录数量
        :param logger: Logger 实例，用于记录分析结果
        """
        self.history_size = history_size
        self.cpu_temp_history = deque(maxlen=history_size)
        self.mem_usage_history = deque(maxlen=history_size)
        self.gpu_usage_history = deque(maxlen=history_size)
        self.logger = logger

    def add_data(self, device_status: DeviceStatus):
        """
        添加新的设备状态数据到历史记录中。

        :param device_status: DeviceStatus 实例
        """
        self.cpu_temp_history.append(device_status.cpu_temp)
        self.mem_usage_history.append(device_status.mem_usage)
        self.gpu_usage_history.append(device_status.gpu_usage)
        if self.logger:
            self.logger.debug(f"添加历史数据: {device_status}")

    def calculate_statistics(self, data: deque) -> Tuple[float, float, float]:
        """
        计算数据的平均值、最大值和最小值。

        :param data: 存储数据的 deque
        :return: (平均值, 最大值, 最小值)
        """
        if not data:
            return (0.0, 0.0, 0.0)
        average = sum(data) / len(data)
        maximum = max(data)
        minimum = min(data)
        return (average, maximum, minimum)

    def determine_trend(self, data: deque) -> str:
        """
        判断数据的趋势是上升、下降还是稳定。

        :param data: 存储数据的 deque
        :return: 趋势描述
        """
        if len(data) < 2:
            return "数据不足，无法判断趋势"

        delta = data[-1] - data[0]
        if delta > 1.0:  # 阈值可以根据需求调整
            return "上升趋势"
        elif delta < -1.0:
            return "下降趋势"
        else:
            return "稳定"

    def generate_report(self) -> str:
        """
        生成分析报告，包含各指标的统计数据和趋势。

        :return: 分析报告字符串
        """
        cpu_avg, cpu_max, cpu_min = self.calculate_statistics(self.cpu_temp_history)
        mem_avg, mem_max, mem_min = self.calculate_statistics(self.mem_usage_history)
        gpu_avg, gpu_max, gpu_min = self.calculate_statistics(self.gpu_usage_history)

        cpu_trend = self.determine_trend(self.cpu_temp_history)
        mem_trend = self.determine_trend(self.mem_usage_history)
        gpu_trend = self.determine_trend(self.gpu_usage_history)

        report = (
            "设备状态分析报告:\n"
            f"CPU 温度 - 平均: {cpu_avg:.2f}°C, 最大: {cpu_max:.2f}°C, 最小: {cpu_min:.2f}°C, 趋势: {cpu_trend}\n"
            f"内存使用率 - 平均: {mem_avg:.2f}%, 最大: {mem_max:.2f}%, 最小: {mem_min:.2f}%, 趋势: {mem_trend}\n"
            f"GPU 使用率 - 平均: {gpu_avg:.2f}%, 最大: {gpu_max:.2f}%, 最小: {gpu_min:.2f}%, 趋势: {gpu_trend}\n"
        )

        if self.logger:
            self.logger.info(report)

        return report
```

接下来，修改 Monitor 类，使其能够与 DeviceAnalyzer 集成，定期分析设备状态数据。

```python
import time
import asyncio
import threading
from typing import Optional
from device_monitor.device_status import DeviceStatus
from device_monitor.logger import Logger
from device_monitor.device_analyzer import DeviceAnalyzer  # 导入 DeviceAnalyzer

class Monitor:
    """
    Monitor 类负责管理监控生命周期，协调各个组件工作，实现优雅的启动和停止。
    它与 DeviceStatus 和 Logger 类集成，支持异步监控、可配置的监控间隔，并提供扩展接口。
    """

    def __init__(self, device_status: 'DeviceStatus', logger: 'Logger', monitor_interval: int = 5, analyzer: Optional['DeviceAnalyzer'] = None):
        """
        初始化 Monitor 类。

        :param device_status: DeviceStatus 实例，用于获取设备状态
        :param logger: Logger 实例，用于记录日志
        :param monitor_interval: 监控间隔（单位：秒）
        :param analyzer: 可选的 DeviceAnalyzer 实例，用于分析设备状态
        """
        self.device_status = device_status
        self.logger = logger
        self.monitor_interval = monitor_interval
        self.analyzer = analyzer
        self._stop_flag = threading.Event()
        self._monitor_thread = None

    async def _monitor_device(self):
        """异步监控设备状态并记录日志。"""
        while not self._stop_flag.is_set():
            try:
                # 检查设备健康状态
                health_status = self.device_status.check_health()

                # 记录设备状态
                self.logger.info(f"设备状态: {health_status}")
                self.logger.debug(f"设备详情: {str(self.device_status)}")

                # 如果存在分析器，添加数据并生成报告
                if self.analyzer:
                    self.analyzer.add_data(self.device_status)
                    report = self.analyzer.generate_report()
                    self.logger.info(report)

                # 按照指定间隔暂停
                await asyncio.sleep(self.monitor_interval)
            except Exception as e:
                # 处理监控过程中出现的任何异常
                self.logger.error(f"监控过程中发生错误: {e}")
                await asyncio.sleep(5)  # 错误恢复等待时间

    def _start_monitoring(self):
        """启动监控线程，执行异步监控任务。"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._monitor_device())

    def start(self):
        """启动监控系统，开始监控设备状态。"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self.logger.warning("监控系统已经在运行中！")
            return

        self._stop_flag.clear()
        self._monitor_thread = threading.Thread(target=self._start_monitoring)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        self.logger.info("监控系统已启动。")

    def stop(self):
        """停止监控系统，结束监控任务。"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self.logger.warning("监控系统未运行！")
            return

        self._stop_flag.set()
        self._monitor_thread.join()
        self.logger.info("监控系统已停止。")

    def restart(self):
        """重启监控系统。"""
        self.stop()
        time.sleep(2)  # 等待一段时间确保系统已经停止
        self.start()
        self.logger.info("监控系统已重启。")

    def get_status(self) -> str:
        """获取当前监控系统的状态。
        :return: 当前监控系统的状态
        """
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            return "监控系统未启动"
        return "监控系统正在运行"

    def set_monitor_interval(self, interval: int):
        """设置监控间隔。
        :param interval: 监控间隔（单位：秒）
        """
        if interval <= 0:
            self.logger.warning("监控间隔必须为正数！")
            return
        self.monitor_interval = interval
        self.logger.info(f"监控间隔已设置为 {interval} 秒。")

    def extend_monitoring(self, additional_monitor: Optional['Monitor'] = None):
        """提供扩展接口，可以添加其他监控任务。
        :param additional_monitor: 可选的扩展监控实例
        """
        if additional_monitor:
            self.logger.info(f"扩展监控系统，添加额外监控任务: {additional_monitor.get_status()}")
            # 这里可以添加逻辑来协调多个监控任务
            additional_monitor.start()
```

最后，在主程序 main.py 中初始化 DeviceAnalyzer 并将其传递给 Monitor 类。

```python
import sys
import signal
import argparse
import logging
import traceback
from time import sleep
from typing import Optional
from device_monitor.monitor import Monitor
from device_monitor.device_status import DeviceStatus
from device_monitor.logger import Logger
from device_monitor.device_analyzer import DeviceAnalyzer  # 导入 DeviceAnalyzer

def setup_logging():
    """
    设置日志记录器，支持多级别日志记录并输出到控制台和文件。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 控制台日志处理器，确保编码为 utf-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件日志处理器，确保编码为 utf-8
    file_handler = logging.FileHandler("monitoring_system.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def handle_exit(signal_number, frame):
    """
    捕获退出信号（如 Ctrl+C）并优雅地停止监控系统。
    """
    print("\n收到退出信号，正在停止监控系统...")
    global monitor_system
    monitor_system.stop()
    print("监控系统已停止，程序退出。")
    sys.exit(0)

def parse_arguments():
    """
    处理命令行参数，支持设置监控间隔。
    """
    parser = argparse.ArgumentParser(description="监控系统 - 启动监控设备状态并记录日志。")
    parser.add_argument(
        "--interval", type=int, default=5, help="设置监控间隔（单位：秒），默认值为 5 秒。"
    )
    parser.add_argument(
        "--history_size", type=int, default=60, help="设置历史数据记录大小，默认值为 60 条。"
    )
    return parser.parse_args()

def main():
    global monitor_system

    # 捕获退出信号（Ctrl+C）
    signal.signal(signal.SIGINT, handle_exit)

    # 解析命令行参数
    args = parse_arguments()

    # 设置日志记录器
    logger = setup_logging()

    # 初始化 DeviceStatus 和 Logger 类
    device_status = DeviceStatus(cpu_temp=55.0, mem_usage=70.0, gpu_usage=60.0)
    logger_instance = Logger(log_file="monitoring_system.log")

    # 初始化 DeviceAnalyzer
    analyzer = DeviceAnalyzer(history_size=args.history_size, logger=logger_instance)

    # 创建并启动 Monitor 系统，传入 analyzer
    monitor_system = Monitor(device_status, logger_instance, monitor_interval=args.interval, analyzer=analyzer)

    try:
        # 启动监控系统
        monitor_system.start()
        print(f"监控系统已启动，监控间隔为 {args.interval} 秒，历史记录大小为 {args.history_size} 条。")

        # 让程序保持运行，直到用户按下 Ctrl+C
        while True:
            sleep(1)

    except Exception as e:
        logger.error(f"程序运行过程中发生错误：{str(e)}")
        print(f"程序运行过程中发生错误，请检查日志文件。")
        traceback.print_exc()
    finally:
        # 确保资源正确释放
        print("正在释放资源...")
        monitor_system.stop()
        print("所有资源已释放，程序结束。")

if __name__ == "__main__":
    main()
```

</details>
