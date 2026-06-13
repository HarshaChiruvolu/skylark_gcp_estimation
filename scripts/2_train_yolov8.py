from ultralytics import YOLO

def main():
    print("Loading YOLOv8 Nano Pose model...")
    # We use the 'nano' version (n) because it trains the fastest and uses the least memory, 
    # but still performs exceptionally well on standard pose tasks.
    model = YOLO('yolov8n-pose.pt')

    print("Starting training...")
    # imgsz=1024 preserves the high resolution required to find the tiny GCP markers
    # device=0 ensures it uses the GPU
    results = model.train(
        data='data/dataset.yaml', 
        epochs=30,               # 30 epochs is usually enough for a fine-tune
        imgsz=1024,              # High res for small objects
        batch=4,                 # Keep batch size small so Colab doesn't run out of memory
        device=0,                # Use GPU
        plots=True               # Generate validation plots
    )
    print("Training complete! Weights saved in runs/pose/train/weights/")

if __name__ == "__main__":
    main()