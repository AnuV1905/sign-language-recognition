import os
import shutil
import random

# CONFIGURATION
src_base = 'data2/raw/original_images'
dst_base = 'data2'
train_split = 0.8  # 80% train, 20% test
delete_raw_after_split = True  # set to False if you want to keep raw images

total_summary = {}

# Loop through all labels (folders like A, B, C...)
for label in os.listdir(src_base):
    src_path = os.path.join(src_base, label)
    if not os.path.isdir(src_path):
        continue

    images = [img for img in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, img))]
    random.shuffle(images)

    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    total_summary[label] = {
        'total': len(images),
        'train': len(train_images),
        'test': len(test_images)
    }

    for phase, image_list in [('train', train_images), ('test', test_images)]:
        dst_path = os.path.join(dst_base, phase, label)
        os.makedirs(dst_path, exist_ok=True)
        for img in image_list:
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(dst_path, img)
            )

# Print summary
print("✅ Data split complete!\n")
print("{:<8} {:<8} {:<8} {:<8}".format("Class", "Total", "Train", "Test"))
print("-" * 32)
for label, stats in total_summary.items():
    print("{:<8} {:<8} {:<8} {:<8}".format(label, stats['total'], stats['train'], stats['test']))

# Optionally delete raw folder
if delete_raw_after_split:
    shutil.rmtree(src_base)
    print("\n🗑️  'raw' folder deleted after split.")
else:
    print("\n📁 'raw' folder retained.")
