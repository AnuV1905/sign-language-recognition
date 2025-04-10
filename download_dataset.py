import kagglehub
import shutil
import os

# Download the dataset (it will be a .zip file)
path = kagglehub.dataset_download("atharvadumbre/indian-sign-language-islrtc-referred")

# Print path to confirm
print("✅ Dataset downloaded to:", path)

# Set up raw folder path
raw_folder = "data2/raw"
os.makedirs(raw_folder, exist_ok=True)

# Extract the dataset contents into the raw folder
for item in os.listdir(path):
    src_item = os.path.join(path, item)
    dst_item = os.path.join(raw_folder, item)

    # Move directories or files into raw/
    if os.path.isdir(src_item):
        shutil.move(src_item, dst_item)
    else:
        shutil.move(src_item, raw_folder)

print("📁 Dataset moved to:", raw_folder)
# trying again