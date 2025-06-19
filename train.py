from ultralytics import YOLO
import torch

def main():
    # Simple GPU check - use GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model and train
    model = YOLO("yolov8s.pt")  # Using yolov8s for better accuracy than yolov8n
    model.train(
        data="dataset/helmet-detection.yaml",
        epochs=100,              # More epochs for better learning
        imgsz=640,               # Keep current size
        batch=8,                 # Lower batch size for less RAM usage
        device=device,
        patience=20,             # Early stopping to prevent overfitting
        lr0=0.01,                # Initial learning rate
        lrf=0.01,                # Final learning rate factor
        weight_decay=0.0005,     # Weight decay for regularization
        warmup_epochs=3,         # Warmup epochs
        save_period=10,          # Save every 10 epochs
        val=True,                # Validate during training
        plots=True,              # Save training plots
        project="runs/train",    # Save results to project/name
        name="helmet_detection", # Save results to project/name
        exist_ok=True,           # Existing project/name ok, do not increment
        workers=2,               # Use 2 workers for better stability on Windows
    )

if __name__ == "__main__":
    main()
