import torch
import psutil
import platform
import time

def check_hardware():
    """Check hardware specifications"""
    print("=== Hardware Analysis ===")
    
    # CPU Info
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"CPU Physical Cores: {psutil.cpu_count(logical=False)}")
    
    # RAM Info
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.total / (1024**3):.1f} GB")
    print(f"Available RAM: {ram.available / (1024**3):.1f} GB")
    
    # GPU Info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\nGPU(s) Found: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Check available GPU memory
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            free = gpu_memory - cached
            print(f"GPU {i} Available Memory: {free:.1f} GB")
    else:
        print("\n❌ No GPU detected - Training will be on CPU")
    
    return torch.cuda.is_available()

def estimate_training_times():
    """Estimate training times for different configurations"""
    print("\n=== Training Time Estimates ===")
    
    # Base estimates (rough approximations)
    # These are estimates for 100 epochs with your dataset size (~1450 images)
    
    configs = [
        {
            "name": "YOLOv8n (nano)",
            "model": "yolov8n.pt",
            "imgsz": 640,
            "batch": 8,
            "cpu_time": "2-3 hours",
            "gpu_time": "30-45 minutes"
        },
        {
            "name": "YOLOv8s (small) - Current",
            "model": "yolov8s.pt", 
            "imgsz": 640,
            "batch": 16,
            "cpu_time": "4-6 hours",
            "gpu_time": "45-60 minutes"
        },
        {
            "name": "YOLOv8m (medium)",
            "model": "yolov8m.pt",
            "imgsz": 640,
            "batch": 16,
            "cpu_time": "8-12 hours",
            "gpu_time": "1-1.5 hours"
        },
        {
            "name": "YOLOv8s + Larger Images",
            "model": "yolov8s.pt",
            "imgsz": 832,
            "batch": 16,
            "cpu_time": "6-8 hours",
            "gpu_time": "1-1.25 hours"
        },
        {
            "name": "YOLOv8m + Larger Images",
            "model": "yolov8m.pt",
            "imgsz": 832,
            "batch": 16,
            "cpu_time": "12-16 hours",
            "gpu_time": "1.5-2 hours"
        }
    ]
    
    has_gpu = torch.cuda.is_available()
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Model: {config['model']}")
        print(f"  Image Size: {config['imgsz']}")
        print(f"  Batch Size: {config['batch']}")
        
        if has_gpu:
            print(f"  Estimated GPU Time: {config['gpu_time']}")
        else:
            print(f"  Estimated CPU Time: {config['cpu_time']}")
        
        # Memory requirements
        if config['model'] == 'yolov8m.pt':
            print(f"  GPU Memory Required: ~4-6 GB")
        elif config['model'] == 'yolov8s.pt':
            print(f"  GPU Memory Required: ~2-4 GB")
        else:
            print(f"  GPU Memory Required: ~1-2 GB")

def recommend_config():
    """Recommend best configuration based on hardware"""
    print("\n=== Recommendations ===")
    
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory >= 6:
            print("✅ Recommended: YOLOv8m with 832px images")
            print("   - Best accuracy")
            print("   - ~1.5-2 hours training time")
        elif gpu_memory >= 4:
            print("✅ Recommended: YOLOv8s with 832px images")
            print("   - Good accuracy")
            print("   - ~1-1.25 hours training time")
        else:
            print("✅ Recommended: YOLOv8s with 640px images")
            print("   - Good accuracy")
            print("   - ~45-60 minutes training time")
    else:
        print("⚠️  CPU Training - Consider using GPU for faster training")
        print("✅ Recommended: YOLOv8s with 640px images")
        print("   - ~4-6 hours training time")

if __name__ == "__main__":
    has_gpu = check_hardware()
    estimate_training_times()
    recommend_config() 