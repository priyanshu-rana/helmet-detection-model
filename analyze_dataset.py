import os
from collections import Counter
import glob

# Paths
labels_dir = "dataset/labels"
images_dir = "dataset/images"

# Count images and labels
image_files = glob.glob(os.path.join(images_dir, "*.png"))
label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

print(f"Total images: {len(image_files)}")
print(f"Total label files: {len(label_files)}")

# Analyze class distribution
class_counts = Counter()
empty_images = 0

for label_file in label_files:
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        if not lines or all(line.strip() == '' for line in lines):
            empty_images += 1
            continue
            
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:  # YOLO format: class x_center y_center width height
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    
    except Exception as e:
        print(f"Error reading {label_file}: {e}")

print(f"\nClass Distribution:")
print(f"Class 0 (With Helmet): {class_counts[0]} instances")
print(f"Class 1 (Without Helmet): {class_counts[1]} instances")
print(f"Empty images: {empty_images}")

# Calculate ratios
total_annotations = sum(class_counts.values())
if total_annotations > 0:
    helmet_ratio = class_counts[0] / total_annotations * 100
    no_helmet_ratio = class_counts[1] / total_annotations * 100
    
    print(f"\nRatios:")
    print(f"With Helmet: {helmet_ratio:.1f}%")
    print(f"Without Helmet: {no_helmet_ratio:.1f}%")
    
    # Simple imbalance check
    if helmet_ratio > 70:
        print("⚠️  WARNING: Dataset imbalanced towards 'With Helmet'")
    elif no_helmet_ratio > 70:
        print("⚠️  WARNING: Dataset imbalanced towards 'Without Helmet'")
    else:
        print("✅ Dataset balance looks good.") 