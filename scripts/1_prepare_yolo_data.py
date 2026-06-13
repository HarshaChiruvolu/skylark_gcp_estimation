import json
import os
import shutil
import random

# ==========================================
# CONFIGURATION
# ==========================================
JSON_PATH = 'data/gcp_marks.json'
RAW_IMAGES_DIR = 'data/' # Assuming you extracted the dataset folders here
YOLO_DIR = 'data/yolo_dataset'

# Image dimensions based on assignment PDF
IMG_WIDTH = 2048.0
IMG_HEIGHT = 1365.0
BOX_SIZE = 100.0 # 100x100 pixel synthetic bounding box

# Class mapping (Fixing the L-Shape vs L-Shaped inconsistency)
CLASS_MAP = {
    "Cross": 0,
    "Square": 1,
    "L-Shape": 2
}

# ==========================================
# SETUP DIRECTORIES
# ==========================================
def create_dirs():
    dirs = [
        f'{YOLO_DIR}/images/train', f'{YOLO_DIR}/images/val',
        f'{YOLO_DIR}/labels/train', f'{YOLO_DIR}/labels/val'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ==========================================
# PROCESS DATA
# ==========================================
def main():
    print("Initializing YOLO dataset preparation...")
    create_dirs()
    
    # Load JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    # Create an 80/20 Train/Validation split
    items = list(data.items())
    random.seed(42) # For reproducibility
    random.shuffle(items)
    
    split_idx = int(len(items) * 0.8)
    train_items = items[:split_idx]
    val_items = items[split_idx:]
    
    def process_split(split_data, split_name):
        print(f"Processing {len(split_data)} images for {split_name}...")
        success_count = 0
        missing_img_count = 0
        corrupted_label_count = 0 # New counter for the unsanitized data trap
        
        for relative_img_path, info in split_data:
            src_img_path = os.path.join(RAW_IMAGES_DIR, relative_img_path)
            
            if not os.path.exists(src_img_path):
                missing_img_count += 1
                continue
                
            # DEFENSIVE PROGRAMMING: Safely extract data using .get() instead of direct keys
            mark_info = info.get('mark', {})
            x_center = mark_info.get('x')
            y_center = mark_info.get('y')
            shape_name = info.get('verified_shape')
            
            # If ANY of the critical data is missing, skip this image
            if x_center is None or y_center is None or shape_name is None:
                corrupted_label_count += 1
                continue
                
            class_id = CLASS_MAP.get(shape_name)
            if class_id is None:
                corrupted_label_count += 1
                continue # Skip invalid shape strings
                
            # Cast to floats now that we know they exist
            x_center = float(x_center)
            y_center = float(y_center)
                
            # Normalize values
            x_norm = x_center / IMG_WIDTH
            y_norm = y_center / IMG_HEIGHT
            w_norm = BOX_SIZE / IMG_WIDTH
            h_norm = BOX_SIZE / IMG_HEIGHT
            
            visibility = 2
            
            # YOLO Format
            yolo_line = f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {x_norm:.6f} {y_norm:.6f} {visibility}\n"
            
            base_name = os.path.basename(src_img_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            dest_img_path = os.path.join(YOLO_DIR, 'images', split_name, base_name)
            dest_label_path = os.path.join(YOLO_DIR, 'labels', split_name, f"{name_without_ext}.txt")
            
            shutil.copy2(src_img_path, dest_img_path)
            with open(dest_label_path, 'w') as label_file:
                label_file.write(yolo_line)
                
            success_count += 1
            
        print(f"{split_name} split done! Copied: {success_count} | Missing Images: {missing_img_count} | Corrupted Labels Skipped: {corrupted_label_count}")

    process_split(train_items, 'train')
    process_split(val_items, 'val')
    
    # ==========================================
    # CREATE YAML FILE FOR TRAINING
    # ==========================================
    yaml_content = f"""
path: {os.path.abspath(YOLO_DIR)} # Absolute path
train: images/train
val: images/val

# Classes
names:
  0: Cross
  1: Square
  2: L-Shaped

# Keypoints
kpt_shape: [1, 3] # 1 keypoint, 3 values (x, y, visibility)
"""
    yaml_path = os.path.join('data', 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
        
    print(f"\nDataset generation complete! YAML file saved to {yaml_path}")

if __name__ == "__main__":
    main()