import os
import shutil
import random

# CONFIGURATION
src_base = 'data2/raw'
dst_base = 'data2'
train_split = 0.8  # 80% train, 20% test

# Loop through all labels (folders like A, B, C...)
for label in os.listdir(src_base):
    src_path = os.path.join(src_base, label)
    if not os.path.isdir(src_path):
        continue

    images = os.listdir(src_path)
    random.shuffle(images)

    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    for phase, image_list in [('train', train_images), ('test', test_images)]:
        dst_path = os.path.join(dst_base, phase, label)
        os.makedirs(dst_path, exist_ok=True)
        for img in image_list:
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(dst_path, img)
            )

print("✅ Successfully split data into train and test sets!")
