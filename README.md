# Helmet Detection System

Detect helmets in images and video using YOLOv8. This project is beginner-friendly and quick to set up.

---

## 📦 Requirements

```bash
pip install ultralytics opencv-python numpy
```

---

## 🚀 Quick Start

1. **Clone this repo and set up your environment:**
   ```bash
   git clone <your-repo-url>
   cd helmet-detection-system
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On Linux/Mac:
   source myenv/bin/activate
   pip install ultralytics opencv-python numpy
   ```
2. **Download and prepare your dataset:**
   - See [Kaggle Helmet Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
   - Place images in `dataset/images/`
   - If you have XML annotations, place them in `dataset/annotations/` and run:
     ```bash
     python xml_to_yolo.py
     ```
   - YOLO TXT labels should be in `dataset/labels/`
   - Update `dataset/helmet-detection.yaml` if you change folder or class names.
3. **Train your model:**
   ```bash
   python train.py
   ```
4. **Run live detection (webcam):**
   ```bash
   python live_detection.py
   ```

---

## 🗂️ Project Structure

```
helmet-detection-system/
├── dataset/
│   ├── helmet-detection.yaml   # Dataset config
│   ├── labels/                # YOLO label files
│   ├── images/                # Image files
│   ├── annotations/           # (Optional) XML annotation files
├── train.py                   # Training script
├── xml_to_yolo.py             # Converts XML to YOLO labels
├── live_detection.py          # Live webcam detection
├── fix_labels.py              # Cleans corrupt YOLO label files
├── README.md
```

---

## ⚠️ Troubleshooting & FAQ

- **Corrupt Label Warnings:**
  If you see warnings about "corrupt image/label" during training, run:

  ```bash
  python fix_labels.py
  ```

  This will remove or fix label files with out-of-bounds or non-normalized coordinates.

- **Windows Multiprocessing Error:**
  If you get a multiprocessing error, make sure your `train.py` code is inside:

  ```python
  if __name__ == "__main__":
      main()
  ```

  and try lowering `workers` to 2 or 0 in `train.py`.

- **GPU Monitoring:**
  Use `nvidia-smi` in a separate terminal to check if your GPU is being used during training. Task Manager may not show correct GPU utilization for deep learning workloads.

- **Model Weights:**
  After training, your best model weights will be saved in `runs/train/helmet_detection/weights/best.pt`.  
  If you move or rename this file, update the path in `live_detection.py`.

- **Dataset Config:**
  If you change folder names or add more classes, update `dataset/helmet-detection.yaml` to match your setup.

---

## ⚙️ Recommended `workers` and `batch` Settings

| Device Type                  | CPU Cores | RAM     | GPU (VRAM) | Recommended `workers` | Recommended `batch` |
| ---------------------------- | --------- | ------- | ---------- | --------------------- | ------------------- |
| Low-end Laptop               | 2-4       | <8 GB   | None/2GB   | 0                     | 4                   |
| Mid-range Laptop/Desktop     | 4-8       | 8-16 GB | 2-4 GB     | 2                     | 8                   |
| High-end Desktop/Workstation | 8+        | 16+ GB  | 6+ GB      | 4-8                   | 16-32               |

**Notes:**

- On **Windows**, if you get a multiprocessing error, lower `workers` to 0 or 2.
- If you run out of RAM or get CUDA OOM errors, lower `batch`.
- For best performance, set `workers` to about half the number of CPU cores, but not more than 8.
- If unsure, start with `workers=2` and `batch=8`.

**Example in `train.py`:**

```python
model.train(
    ...,
    batch=8,      # Adjust as per your GPU RAM
    workers=2,    # Adjust as per your CPU cores and OS
    ...
)
```

**Auto-detect workers in `train.py` (optional):**

```python
import os
cpu_cores = os.cpu_count()
if cpu_cores <= 4:
    workers = 0
elif cpu_cores <= 8:
    workers = 2
else:
    workers = 4

model.train(
    ...,
    workers=workers,
    ...
)
```

---

## 📚 Useful Links

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Helmet Detection Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)

---

## 🤝 Contributing

Found a bug or have an idea? Open an issue or pull request. Happy coding!
