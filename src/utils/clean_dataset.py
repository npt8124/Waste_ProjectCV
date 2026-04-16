import os

def clean_labels(label_dir, img_dir):
    removed = 0
    fixed = 0

    for file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, file)
        img_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()

            # chỉ giữ đúng format YOLO
            if len(parts) == 5:
                try:
                    float_vals = list(map(float, parts))
                    new_lines.append(line)
                except:
                    continue
            else:
                fixed += 1

        # nếu label rỗng → xoá cả ảnh + label
        if len(new_lines) == 0:
            os.remove(label_path)
            if os.path.exists(img_path):
                os.remove(img_path)
            removed += 1
        else:
            with open(label_path, "w") as f:
                f.writelines(new_lines)

    print(f"Removed files: {removed}")
    print(f"Fixed lines: {fixed}")


if __name__ == "__main__":
    clean_labels(
        "datasets/detect/train/labels",
        "datasets/detect/train/images"
    )