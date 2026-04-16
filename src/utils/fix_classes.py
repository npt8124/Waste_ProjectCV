import os

def fix_labels(label_dir):
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) == 5:
                cls = int(parts[0])

                # chỉ giữ garbage (class 1 → đổi thành 0)
                if cls == 1:
                    parts[0] = "0"
                    new_lines.append(" ".join(parts) + "\n")

        # nếu file rỗng → xoá
        if len(new_lines) == 0:
            os.remove(path)
        else:
            with open(path, "w") as f:
                f.writelines(new_lines)


if __name__ == "__main__":
    fix_labels("datasets/detect/train/labels")
    fix_labels("datasets/detect/valid/labels")