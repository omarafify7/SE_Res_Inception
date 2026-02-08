
import io
import os
import zipfile
import requests
import shutil
from tqdm import tqdm

DATA_DIR = './data'
URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
TARGET_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200')

def download_and_extract():
    if os.path.exists(TARGET_DIR):
        print(f"Dataset already exists at {TARGET_DIR}")
        return

    print(f"Downloading Tiny ImageNet from {URL}...")
    response = requests.get(URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Download to memory buffer
    buffer = io.BytesIO()
    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for data in response.iter_content(1024):
            buffer.write(data)
            pbar.update(len(data))
            
    print("Extracting...")
    with zipfile.ZipFile(buffer) as zf:
        zf.extractall(DATA_DIR)
        
    print("Download complete.")

def restructure_val():
    """
    Restructures the validation set from:
    val/
        images/
            val_0.JPEG
        val_annotations.txt
        
    to:
    val/
        n01440764/
            val_0.JPEG
        ...
    """
    val_dir = os.path.join(TARGET_DIR, 'val')
    img_dir = os.path.join(val_dir, 'images')
    anno_file = os.path.join(val_dir, 'val_annotations.txt')
    
    if not os.path.exists(img_dir):
        print("Validation folder already structured or missing.")
        return

    print("Restructuring validation set...")
    
    # Read annotations
    with open(anno_file, 'r') as f:
        lines = f.readlines()
        
    # Create valid dict: filename -> class_id
    val_img_dict = {}
    for line in lines:
        parts = line.strip().split('\t')
        val_img_dict[parts[0]] = parts[1]

    # Move images
    for img_file, class_id in val_img_dict.items():
        src = os.path.join(img_dir, img_file)
        dst_folder = os.path.join(val_dir, class_id)
        
        os.makedirs(dst_folder, exist_ok=True)
        
        # Check if file exists in source (it might have been moved already)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dst_folder, img_file))
            
    # Cleanup
    if os.path.exists(img_dir) and not os.listdir(img_dir):
        os.rmdir(img_dir)
        
    print("Validation set restructured.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_and_extract()
    restructure_val()
