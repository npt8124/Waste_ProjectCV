import os

def check_labels(label_dir):
    total = 0
    empty = 0
    wrong = 0

    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                wrong += 1

        total += 1

    print(f"Total labels: {total}")
    print(f"Empty labels: {empty}")
    print(f"Wrong format: {wrong}")


if __name__ == "__main__":
    check_labels("datasets/detect/train/labels")
    check_labels("datasets/detect/valid/labels")