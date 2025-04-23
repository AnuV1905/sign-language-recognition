import numpy as np
import cv2
import os
import csv
from image_processing import func

# Create necessary directories if they don't exist
if not os.path.exists("data2"):
    os.makedirs("data2")
if not os.path.exists("data2/train"):
    os.makedirs("data2/train")
if not os.path.exists("data2/test"):
    os.makedirs("data2/test")

path = "train"
path1 = "data2"
a = ['label']

for i in range(64 * 64):
    a.append("pixel" + str(i))

label = 0
var = 0
c1 = 0
c2 = 0

for (dirpath, dirnames, filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for (direcpath, direcnames, files) in os.walk(os.path.join(path, dirname)):
            train_dir = os.path.join(path1, "train", dirname)
            test_dir = os.path.join(path1, "test", dirname)

            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            # For all files, process and write to train/test
            num = 100000000000000000  # Artificially large to ensure all go to train
            i = 0
            for file in files:
                var += 1
                actual_path = os.path.join(path, dirname, file)
                actual_path1 = os.path.join(train_dir, file)
                actual_path2 = os.path.join(test_dir, file)

                img = cv2.imread(actual_path, 0)
                bw_image = func(actual_path)
                bw_image = cv2.resize(bw_image, (128, 128))

                if i < num:
                    c1 += 1
                    cv2.imwrite(actual_path1, bw_image)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2, bw_image)

                i += 1

        label += 1

print("Total Images Processed:", var)
print("Training Images:", c1)
print("Testing Images:", c2)
