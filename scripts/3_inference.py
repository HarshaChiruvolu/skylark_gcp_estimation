import json
import glob
import os
from ultralytics import YOLO

def main():
    print("Loading trained model weights...")
    model = YOLO('weights/best.pt')

    test_dir = 'data/test_dataset'
    test_images_path = os.path.join(test_dir, '**', '*.JPG')
    test_images = glob.glob(test_images_path, recursive=True)

    if not test_images:
        print(f"No test images found! Please check {test_dir}")
        return

    predictions = {}
    class_names = {0: "Cross", 1: "Square", 2: "L-Shaped"}

    print(f"Running high-speed inference on {len(test_images)} images...")

    fallback_count = 0

    for img_path in test_images:
        # Removing augment=True to fix the CPU hang/slowdown
        # Using a ultra-low confidence threshold of 0.001 to maximize catch rate
        results = model(img_path, imgsz=1024, verbose=False, conf=0.001)
        
        # Format the path for the JSON key
        relative_path = os.path.relpath(img_path, 'data').replace('\\', '/')

        if len(results[0].boxes) > 0 and results[0].keypoints is not None:
            # Extract highest confidence prediction
            cls_id = int(results[0].boxes.cls[0].item())
            keypoints = results[0].keypoints.xy[0][0] 

            predictions[relative_path] = {
                "mark": {
                    "x": float(keypoints[0].item()),
                    "y": float(keypoints[1].item())
                },
                "verified_shape": class_names.get(cls_id, "Cross")
            }
        else:
            # FALLBACK SYSTEM: Guarantees the evaluation script won't crash on missing entries
            fallback_count += 1
            predictions[relative_path] = {
                "mark": {
                    "x": 1024.0,  # Center of a 2048-width image
                    "y": 682.5   # Center of a 1365-height image
                },
                "verified_shape": "Cross"  # Fallback to the most dominant class
            }

    output_file = 'predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"\nSuccess! Processed all {len(test_images)} images.")
    print(f"Saved complete results to {output_file}")
    if fallback_count > 0:
        print(f"Note: Applied safe center-fallback for {fallback_count} undetected images.")

if __name__ == "__main__":
    main()