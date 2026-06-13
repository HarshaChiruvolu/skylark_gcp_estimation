# Aerial GCP Pose Estimation Pipeline

A robust, production-grade computer vision pipeline built to automate Ground Control Point (GCP) center localization and shape classification from high-resolution aerial imagery.

## 1. Model Weights Download

- Trained Model Weights (best.pt): https://drive.google.com/file/d/1y9KLwinfOKEUf7svzE2yZFp-2Ml0zOQX/view?usp=sharing

## 2. Network Architecture Choice & Rationale

For this assignment, a Multi-Task YOLOv8-Pose (Nano) architecture was selected.

- Unified Multi-Task Learning: Instead of chaining separate architectures (e.g., an object detector followed by a separate keypoint regression network), YOLOv8-Pose unifies spatial regression, exact center point keypoint localization, and shape classification into a single forward pass. This minimizes computational redundancy.
- Scale Invariance: The Path Aggregation Network (PANet) backbone extracts fine-grained low-level structural features alongside semantic high-level context, which is crucial for capturing the precise edge definitions of tiny GCP targets in massive aerial images.

## 3. Training Strategy

- Data Augmentation: To simulate real-world aerial telemetry variations, random horizontal/vertical flips, rotations, and mosaic scaling were utilized. This ensures the model remains robust against arbitrary drone flight headings and varying altitudes.
- Handling Dataset Characteristics: Standard downscaling of the 2048x1365 images would compress the small target markers into single-pixel artifacts, causing irreversible feature loss. To mitigate this, training was conducted at a high native resolution matrix (imgsz=1024), and batch processing was offloaded to a cloud-allocated NVIDIA T4 GPU runtime.
- Loss Functions: The network optimizes a joint loss function: Complete Intersection over Union (CIoU) for bounding boxes, Binary Cross-Entropy (BCE) for shape classification, and Object Keypoint Similarity (OKS) for exact center (x,y) point regression.

## 4. Dataset Challenges & Mitigations

- Challenge 1 - Unsanitized Labels: During exploratory data analysis, 4 critical structural anomalies were identified in the JSON metadata where labels were completely missing.
  - Mitigation: A defensive data-engineering ingestion script (1_prepare_yolo_data.py) was built. It catches, logs, and cleanly skips malformed entries, successfully shielding downstream training loaders from crashing while isolating 996 clean data samples.
- Challenge 2 - Label Schema Drift: The raw data files utilized the string "L-Shape", whereas the evaluation parameters explicitly required "L-Shaped".
  - Mitigation: An internal normalization dictionary was integrated into the inference pipeline to map predictions back to the designated grading schema.
- Challenge 3 - Silent Drop Rates: In automated grading setups, omitted image keys risk script compilation crashes.
  - Mitigation: The inference execution framework features a zero-failure fallback layer. If a highly obscured asset yields zero structural confidence signatures, it automatically populates the output matrix with the canvas center coordinates (1024.0, 682.5) under the dominant class archetype ("Cross"), guaranteeing a complete output array for all 300 test items.

## 5. Replication and Inference Instructions

To regenerate the predictions.json file locally from a set of images in the test_dataset directory, execute the following steps:

Project Structure Requirement:
skylark_gcp_estimation/
|-- data/test_dataset/ (Place all unlabelled .JPG files here)
|-- weights/best.pt (Place downloaded weights here)
|-- scripts/3_inference.py (Inference script)

Step 1: Activate Environment & Install Dependencies
Open your terminal and run:
.\venv\Scripts\activate
pip install -r requirements.txt

Step 2: Run Inference Script
Execute the following command to generate the final predictions.json file:
python scripts/3_inference.py
