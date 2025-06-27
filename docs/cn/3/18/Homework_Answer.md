# 课后拓展参考答案

## 练习1：自定义数据集训练与评估

**任务描述**：

+ 使用 YOLOv8 训练一个自定义的目标检测模型，体验完整的训练流程。
+ 实现数据集的收集、标注、训练和评估全过程。
+ 对比不同模型配置对检测性能的影响。
+ 可视化训练过程并分析模型性能指标。

**提示**：

+ 选择一个具体的检测任务，如检测特定物品、动物或车辆类型。
+ 使用 LabelImg 或 Roboflow 等工具进行数据标注。
+ 利用数据增强技术扩充训练数据。
+ 监控训练过程中的损失变化和 mAP 指标。

<details>
<summary>点击查看答案</summary>

```python
import torch
from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class CustomDatasetTrainer:
    def __init__(self, project_name="custom_detection"):
        """自定义数据集训练器"""
        self.project_name = project_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"训练设备: {self.device}")
        
        # 创建项目目录结构
        self.setup_project_structure()
    
    def setup_project_structure(self):
        """创建项目目录结构"""
        self.project_dir = Path(self.project_name)
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        
        # 创建目录
        for dir_path in [self.project_dir, self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # 创建数据子目录
        for split in ['train', 'val', 'test']:
            for folder in ['images', 'labels']:
                (self.data_dir / split / folder).mkdir(exist_ok=True, parents=True)
        
        print(f"项目目录结构已创建: {self.project_dir}")
    
    def create_dataset_config(self, class_names):
        """创建数据集配置文件"""
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        config_path = self.data_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"数据集配置已保存: {config_path}")
        return config_path
    
    def generate_demo_dataset(self, num_samples_per_class=50):
        """生成演示数据集（用于测试训练流程）"""
        print("生成演示数据集...")
        
        import cv2
        import random
        
        class_names = ['square', 'circle', 'triangle']
        
        def create_synthetic_image():
            """创建包含几何形状的合成图像"""
            img = np.ones((416, 416, 3), dtype=np.uint8) * 255  # 白色背景
            
            annotations = []
            num_objects = random.randint(1, 3)
            
            for _ in range(num_objects):
                # 随机选择形状类型
                shape_type = random.randint(0, 2)
                
                # 随机位置和大小
                center_x = random.randint(50, 366)
                center_y = random.randint(50, 366)
                size = random.randint(30, 80)
                
                # 随机颜色
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                if shape_type == 0:  # 矩形
                    top_left = (center_x - size//2, center_y - size//2)
                    bottom_right = (center_x + size//2, center_y + size//2)
                    cv2.rectangle(img, top_left, bottom_right, color, -1)
                    
                elif shape_type == 1:  # 圆形
                    cv2.circle(img, (center_x, center_y), size//2, color, -1)
                    
                else:  # 三角形
                    points = np.array([
                        [center_x, center_y - size//2],
                        [center_x - size//2, center_y + size//2],
                        [center_x + size//2, center_y + size//2]
                    ], np.int32)
                    cv2.fillPoly(img, [points], color)
                
                # 计算边界框（YOLO格式：x_center, y_center, width, height，归一化到0-1）
                x_center = center_x / 416
                y_center = center_y / 416
                width = size / 416
                height = size / 416
                
                annotations.append([shape_type, x_center, y_center, width, height])
            
            return img, annotations
        
        # 生成训练和验证数据
        for split in ['train', 'val']:
            samples = num_samples_per_class if split == 'train' else num_samples_per_class // 4
            
            for i in range(samples):
                # 生成图像和标注
                img, annotations = create_synthetic_image()
                
                # 保存图像
                img_filename = f"{split}_{i:04d}.jpg"
                img_path = self.data_dir / split / "images" / img_filename
                cv2.imwrite(str(img_path), img)
                
                # 保存标注
                label_filename = f"{split}_{i:04d}.txt"
                label_path = self.data_dir / split / "labels" / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(' '.join(map(str, ann)) + '\n')
        
        print(f"演示数据集生成完成")
        print(f"  训练集: {num_samples_per_class} 张图像")
        print(f"  验证集: {num_samples_per_class // 4} 张图像")
        
        return class_names
    
    def train_model(self, config_path, model_size='n', epochs=50, imgsz=416, batch_size=16):
        """训练模型"""
        print(f"开始训练YOLOv8{model_size}模型...")
        
        # 加载预训练模型
        model = YOLO(f'yolov8{model_size}.pt')
        
        # 开始训练
        results = model.train(
            data=str(config_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            project=str(self.results_dir),
            name='custom_train',
            save_period=10,  # 每10个epoch保存一次
            plots=True,      # 生成训练图表
            verbose=True
        )
        
        print("训练完成！")
        return model, results
    
    def evaluate_model(self, model_path, config_path):
        """评估模型性能"""
        print("评估模型性能...")
        
        # 加载训练好的模型
        model = YOLO(model_path)
        
        # 在验证集上评估
        metrics = model.val(
            data=str(config_path),
            device=self.device,
            plots=True
        )
        
        # 输出评估结果
        print(f"\n=== 模型评估结果 ===")
        if hasattr(metrics, 'box'):
            print(f"mAP50: {metrics.box.map50:.3f}")
            print(f"mAP50-95: {metrics.box.map:.3f}")
            print(f"Precision: {metrics.box.p.mean():.3f}")
            print(f"Recall: {metrics.box.r.mean():.3f}")
        
        return metrics
    
    def visualize_training_results(self, results_dir):
        """可视化训练结果"""
        print("生成训练结果可视化...")
        
        # 查找训练结果文件
        train_dir = Path(results_dir) / "custom_train"
        
        if not train_dir.exists():
            print("未找到训练结果目录")
            return
        
        # 显示训练图表
        plots = ['results.png', 'confusion_matrix.png', 'val_batch0_pred.jpg']
        
        fig, axes = plt.subplots(1, len(plots), figsize=(15, 5))
        if len(plots) == 1:
            axes = [axes]
        
        for i, plot_name in enumerate(plots):
            plot_path = train_dir / plot_name
            if plot_path.exists():
                img = plt.imread(plot_path)
                axes[i].imshow(img)
                axes[i].set_title(plot_name.split('.')[0].replace('_', ' ').title())
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'{plot_name}\n未找到', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def test_inference(self, model_path, test_images_dir):
        """测试推理性能"""
        print("测试推理性能...")
        
        model = YOLO(model_path)
        
        # 获取测试图像
        test_images = list(Path(test_images_dir).glob("*.jpg"))
        
        if not test_images:
            print("未找到测试图像")
            return
        
        import time
        
        # 预热
        dummy_img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        for _ in range(5):
            _ = model(dummy_img, verbose=False)
        
        # 性能测试
        times = []
        for img_path in test_images[:10]:  # 测试前10张图像
            img = cv2.imread(str(img_path))
            
            start_time = time.time()
            results = model(img, verbose=False)
            inference_time = time.time() - start_time
            
            times.append(inference_time)
            
            # 显示检测结果
            if len(results) > 0 and results[0].boxes is not None:
                detection_count = len(results[0].boxes)
                print(f"{img_path.name}: {detection_count} 个检测, {inference_time*1000:.1f}ms")
        
        if times:
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            print(f"\n平均推理时间: {avg_time*1000:.1f}ms")
            print(f"平均FPS: {fps:.1f}")
    
    def run_complete_training_pipeline(self):
        """运行完整训练流程"""
        print("=== 自定义数据集训练完整流程 ===")
        
        try:
            # 1. 生成演示数据集
            class_names = self.generate_demo_dataset(num_samples_per_class=100)
            
            # 2. 创建数据集配置
            config_path = self.create_dataset_config(class_names)
            
            # 3. 训练模型
            model, results = self.train_model(
                config_path, 
                model_size='n',  # 使用nano模型以适应边缘设备
                epochs=20,       # 减少训练轮数以加快演示
                batch_size=8     # 较小的批次大小
            )
            
            # 4. 评估模型
            best_model_path = self.results_dir / "custom_train" / "weights" / "best.pt"
            if best_model_path.exists():
                metrics = self.evaluate_model(str(best_model_path), config_path)
            
            # 5. 可视化结果
            self.visualize_training_results(self.results_dir)
            
            # 6. 测试推理
            test_images_dir = self.data_dir / "val" / "images"
            if best_model_path.exists():
                self.test_inference(str(best_model_path), str(test_images_dir))
            
            print("\n✅ 训练流程完成！")
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

# 使用示例
if __name__ == "__main__":
    trainer = CustomDatasetTrainer("shape_detection_project")
    trainer.run_complete_training_pipeline()
```

</details>

---

## 练习2：边缘设备部署优化实践

**任务描述**：

+ 实现 YOLOv8 模型在 Jetson 设备上的深度优化。
+ 对比不同优化策略的性能提升效果。
+ 实现模型量化、TensorRT 优化等高级技术。
+ 构建完整的性能监控和分析系统。

**提示**：

+ 使用 TensorRT 进行模型加速优化。
+ 实验不同的输入分辨率和精度设置。
+ 监控 GPU 和 CPU 使用率、温度等系统指标。
+ 建立性能基准测试框架。

<details>
<summary>点击查看答案</summary>

```python
import torch
import time
import psutil
import threading
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class JetsonOptimizer:
    def __init__(self, model_path='yolov8n.pt'):
        """Jetson设备优化器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.optimization_results = {}
        
        print(f"优化目标设备: {self.device}")
        self.check_system_specs()
    
    def check_system_specs(self):
        """检查系统规格"""
        print("\n=== 系统规格检查 ===")
        
        # CPU信息
        cpu_info = {
            'cores': psutil.cpu_count(),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'
        }
        print(f"CPU核心数: {cpu_info['cores']}")
        print(f"CPU频率: {cpu_info['frequency']} MHz")
        
        # 内存信息
        memory = psutil.virtual_memory()
        print(f"总内存: {memory.total / 1024**3:.1f} GB")
        print(f"可用内存: {memory.available / 1024**3:.1f} GB")
        
        # GPU信息
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU内存: {memory_gb:.1f} GB")
            print(f"CUDA版本: {torch.version.cuda}")
        
        return {
            'cpu': cpu_info,
            'memory_gb': memory.total / 1024**3,
            'gpu_memory_gb': memory_gb if torch.cuda.is_available() else 0
        }
    
    def benchmark_baseline_model(self, num_runs=50, input_size=(640, 640)):
        """基准模型性能测试"""
        print(f"\n=== 基准性能测试 (输入尺寸: {input_size}) ===")
        
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            
            # 创建测试输入
            test_input = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)
            
            # 预热GPU
            print("GPU预热中...")
            for _ in range(10):
                _ = model(test_input, verbose=False)
            
            # 性能测试
            print(f"执行{num_runs}次推理测试...")
            times = []
            
            for i in range(num_runs):
                start_time = time.time()
                results = model(test_input, verbose=False)
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                if (i + 1) % 10 == 0:
                    print(f"  完成 {i+1}/{num_runs}")
            
            # 计算统计信息
            times = np.array(times)
            stats = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'fps': 1.0 / np.mean(times),
                'input_size': input_size
            }
            
            print(f"\n基准测试结果:")
            print(f"  平均推理时间: {stats['mean_time']*1000:.1f}ms ± {stats['std_time']*1000:.1f}ms")
            print(f"  最快/最慢: {stats['min_time']*1000:.1f}ms / {stats['max_time']*1000:.1f}ms")
            print(f"  平均FPS: {stats['fps']:.1f}")
            
            self.optimization_results['baseline'] = stats
            return stats
            
        except Exception as e:
            print(f"基准测试失败: {e}")
            return None
    
    def optimize_input_resolution(self):
        """优化输入分辨率"""
        print("\n=== 输入分辨率优化 ===")
        
        resolutions = [
            (320, 320),
            (416, 416),
            (640, 640),
            (800, 800),
            (1024, 1024)
        ]
        
        resolution_results = {}
        
        for res in resolutions:
            print(f"\n测试分辨率 {res[0]}x{res[1]}...")
            stats = self.benchmark_baseline_model(num_runs=20, input_size=res)
            
            if stats:
                resolution_results[f"{res[0]}x{res[1]}"] = stats
                efficiency = stats['fps'] * res[0] * res[1] / 1000000  # FPS * 百万像素
                print(f"  效率指标: {efficiency:.2f}")
        
        self.optimization_results['resolutions'] = resolution_results
        return resolution_results
    
    def optimize_model_precision(self):
        """模型精度优化"""
        print("\n=== 模型精度优化 ===")
        
        precision_configs = [
            {'conf': 0.25, 'iou': 0.45, 'max_det': 300},
            {'conf': 0.4, 'iou': 0.5, 'max_det': 100},
            {'conf': 0.5, 'iou': 0.5, 'max_det': 50},
            {'conf': 0.6, 'iou': 0.6, 'max_det': 30}
        ]
        
        precision_results = {}
        
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            for i, config in enumerate(precision_configs):
                print(f"\n测试配置 {i+1}: conf={config['conf']}, iou={config['iou']}, max_det={config['max_det']}")
                
                times = []
                detection_counts = []
                
                # 预热
                for _ in range(5):
                    _ = model(test_input, **config, verbose=False)
                
                # 测试
                for _ in range(20):
                    start_time = time.time()
                    results = model(test_input, **config, verbose=False)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
                    # 统计检测数量
                    if results and len(results) > 0 and results[0].boxes is not None:
                        detection_counts.append(len(results[0].boxes))
                    else:
                        detection_counts.append(0)
                
                avg_time = np.mean(times)
                avg_detections = np.mean(detection_counts)
                fps = 1.0 / avg_time
                
                config_name = f"conf{config['conf']}_iou{config['iou']}_max{config['max_det']}"
                precision_results[config_name] = {
                    'avg_time': avg_time,
                    'fps': fps,
                    'avg_detections': avg_detections,
                    'config': config
                }
                
                print(f"  平均时间: {avg_time*1000:.1f}ms, FPS: {fps:.1f}, 平均检测数: {avg_detections:.1f}")
            
            self.optimization_results['precision'] = precision_results
            return precision_results
            
        except Exception as e:
            print(f"精度优化测试失败: {e}")
            return None
    
    def generate_optimization_report(self):
        """生成优化报告"""
        print("\n=== 生成优化报告 ===")
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 分辨率对比
        if 'resolutions' in self.optimization_results:
            res_data = self.optimization_results['resolutions']
            resolutions = list(res_data.keys())
            fps_values = [res_data[res]['fps'] for res in resolutions]
            
            axes[0, 0].bar(resolutions, fps_values, color='skyblue')
            axes[0, 0].set_title('不同分辨率的FPS对比')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 精度配置对比
        if 'precision' in self.optimization_results:
            prec_data = self.optimization_results['precision']
            configs = list(prec_data.keys())
            fps_values = [prec_data[config]['fps'] for config in configs]
            
            axes[0, 1].bar(range(len(configs)), fps_values, color='lightgreen')
            axes[0, 1].set_title('不同精度配置的FPS对比')
            axes[0, 1].set_ylabel('FPS')
            axes[0, 1].set_xticks(range(len(configs)))
            axes[0, 1].set_xticklabels([f"Config {i+1}" for i in range(len(configs))], rotation=45)
        
        # 3. 优化效果对比
        optimization_types = []
        fps_improvements = []
        
        baseline_fps = self.optimization_results.get('baseline', {}).get('fps', 0)
        
        if baseline_fps > 0:
            optimization_types.append('Baseline')
            fps_improvements.append(baseline_fps)
        
        if optimization_types:
            axes[1, 0].bar(optimization_types, fps_improvements, color='orange')
            axes[1, 0].set_title('优化方法效果对比')
            axes[1, 0].set_ylabel('FPS')
        
        # 4. 空白区域放置文字总结
        axes[1, 1].axis('off')
        summary_text = """优化总结:

1. 选择合适的输入分辨率
2. 调整检测阈值参数
3. 限制最大检测数量
4. 使用混合精度推理
5. 考虑TensorRT加速

建议配置:
- 分辨率: 416x416
- 置信度: 0.5
- IoU阈值: 0.45
- 最大检测: 100"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # 保存详细报告
        report_path = "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2, default=str)
        
        print(f"详细报告已保存: {report_path}")
    
    def run_complete_optimization(self):
        """运行完整优化流程"""
        print("=== Jetson设备完整优化流程 ===")
        
        try:
            # 1. 基准测试
            self.benchmark_baseline_model()
            
            # 2. 分辨率优化
            self.optimize_input_resolution()
            
            # 3. 精度配置优化
            self.optimize_model_precision()
            
            # 4. 生成报告
            self.generate_optimization_report()
            
            print("\n✅ 完整优化流程完成！")
            
        except Exception as e:
            print(f"❌ 优化过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

# 使用示例
if __name__ == "__main__":
    optimizer = JetsonOptimizer('yolov8n.pt')
    optimizer.run_complete_optimization()
```

</details>
