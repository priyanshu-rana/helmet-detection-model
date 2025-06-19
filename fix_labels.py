import os

labels_dir = 'dataset/labels'
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

fixed = 0
removed = 0
skipped = 0

for fname in label_files:
    path = os.path.join(labels_dir, fname)
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                coords = list(map(float, parts[1:5]))
                if all(0.0 <= c <= 1.0 for c in coords):
                    new_lines.append(line)
            except Exception:
                continue
    if len(new_lines) == 0:
        os.remove(path)
        removed += 1
    elif len(new_lines) < len(lines):
        with open(path, 'w') as f:
            f.writelines(new_lines)
        fixed += 1
    else:
        skipped += 1

print(f"Fixed label files: {fixed}")
print(f"Removed label files: {removed}")
print(f"Unchanged label files: {skipped}")
print("Done.") 