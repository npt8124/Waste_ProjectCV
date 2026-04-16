import os
import shutil
import random

src_dir = "dataset-resized"
out_dir = "datasets/classify"

classes = os.listdir(src_dir)

for cls in classes:
    images = os.listdir(os.path.join(src_dir, cls))
    random.shuffle(images)

    split = int(0.8 * len(images))

    train_imgs = images[:split]
    val_imgs = images[split:]

    for img in train_imgs:
        src = os.path.join(src_dir, cls, img)
        dst = os.path.join(out_dir, "train", cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

    for img in val_imgs:
        src = os.path.join(src_dir, cls, img)
        dst = os.path.join(out_dir, "valid", cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

print("DONE SPLIT")