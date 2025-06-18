# save as xml_to_yolo.py
import os
import xml.etree.ElementTree as ET

# Class mapping (replace with your class names)
classes = {"With Helmet": 0, "Without Helmet": 1}  # Add other classes if needed


def convert_xml_to_txt(xml_path, output_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    txt_lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue  # Skip unknown classes
        cls_id = classes[cls]

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format (normalized center coordinates)
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        txt_lines.append(f"{cls_id} {x_center} {y_center} {w} {h}")

    # Save to labels folder
    filename = os.path.splitext(os.path.basename(xml_path))[0] + ".txt"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("\n".join(txt_lines))


# Run conversion
xml_dir = "./dataset/annotations"
output_dir = "./dataset/labels"
os.makedirs(output_dir, exist_ok=True)

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        convert_xml_to_txt(os.path.join(xml_dir, xml_file), output_dir)
