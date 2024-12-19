# 全局配置变量
DEVICE_ID = "EDGE_AI_001"
MAX_TEMPERATURE = 75
SAMPLING_RATE = 1000

def check_temperature():
    """检查设备温度是否超过全局设置的阈值"""
    current_temp = get_device_temperature()  # 假设有这个函数
    print(f"设备 {DEVICE_ID} 当前温度: {current_temp}°C")
    if current_temp > MAX_TEMPERATURE:
        print(f"警告：温度超过阈值 {MAX_TEMPERATURE}°C")