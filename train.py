from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="dataset/helmet-detection.yaml",  # Updated path
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
)
