# Helmet Detection System

Welcome! This project helps you detect helmets in images and live video using YOLOv8. Whether you're a beginner or just want a quick start, you're in the right place.

---

## 📦 Requirements

Install the required Python packages:
```bash
pip install ultralytics opencv-python numpy
```

---

## 🚦 What Can You Do Here?
- Train a helmet detector on your own images
- Convert XML annotations to YOLO format (so your model understands them)
- Run live helmet detection from your webcam

---

## 🗂️ What's Inside?
```
helmet-detection-system/
├── dataset/
│   ├── helmet-detection.yaml   # Dataset config
├── train.py                   # Training script
├── xml_to_yolo.py             # Converts XML to YOLO labels
├── live_detection.py          # Live webcam detection
├── .gitignore                 
├── README.md                 
```

---

## 🚀 Quick Start

1. **Clone this repo:**
   ```bash
   git clone <your-repo-url>
   cd helmet-detection-system
   ```
2. **Set up your Python environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   pip install ultralytics opencv-python numpy
   ```

---

## 📥 Add Your Dataset (Kaggle Reference)

You can use **any helmet detection dataset** that fits the YOLO format, but for convenience, I recommend the [Helmet Detection dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/helmet-detection). If you want to use this dataset, here's how to get set up:

1. **Download the dataset from Kaggle and unzip it.**
2. **Organize your files like this:**
   - Images: `dataset/images/`
   - XML annotations: `dataset/annotations/` (if you have them)
   - YOLO TXT labels: `dataset/labels/` (if you have them, or after conversion)
3. **If you have XMLs, convert them to YOLO TXT:**
   ```bash
   python xml_to_yolo.py
   ```
   This will create `.txt` label files in `dataset/labels/`.
4. **If you change folder or class names, update `dataset/helmet-detection.yaml`.**

> Need more info? Check the [Kaggle dataset page](https://www.kaggle.com/datasets/andrewmvd/helmet-detection).

---

## 🛠️ Why Do I Need `xml_to_yolo.py`?

- YOLOv8 needs labels in a special TXT format.
- Many datasets (like the one from Kaggle) use XML files for annotations.
- `xml_to_yolo.py` converts those XMLs into the format YOLO understands. Run it once, and you're ready to train!

---

## 🏁 Typical Workflow

```bash
# 1. Download and organize your dataset
# 2. Convert XML to YOLO TXT (if needed)
python xml_to_yolo.py

# 3. Train your model
python train.py

# 4. Try live detection
python live_detection.py
```

---

## ⚠️ Important Notes

- **Model Weights:**  
  After training, my best model weights will be saved in `runs/detect/train2/weights/best.pt`.  
  If you move or rename this file, update the path in `live_detection.py` accordingly.

- **Dataset Config:**  
  If you change folder names or add more classes, update `dataset/helmet-detection.yaml` to match your setup.

---

## 📝 Notes
- This repo skips big files (datasets, weights, outputs) to keep things fast and clean.
- After cloning, just add your dataset and weights as described above.
- Works on Windows or Ubuntu. For GPU speed, install NVIDIA drivers and CUDA for your OS.

---

## 📚 Useful Links
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Helmet Detection Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)

---

## 🤝 Contributing
Found a bug or have an idea? Open an issue or pull request. Happy coding! 