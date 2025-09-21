# Deepfake Video Detection using GAN

## Overview

This repository implements a **deepfake video detection system** using a **Generative Adversarial Network (GAN)**-based approach. It is designed to classify videos as *real* or *fake* by analyzing facial regions and temporal patterns extracted from video frames. The project includes preprocessing scripts, model training, evaluation, and a simple demo web interface for inference.

## Key Features

* Face detection and alignment pipeline for robust frame extraction.
* GAN-based classifier that learns to distinguish real from manipulated faces.
* Support for common deepfake datasets (e.g., FaceForensics++, DFDC) and custom datasets.
* Scripts for training, evaluation, and running inference on single videos.
* Streamlit demo for quick video testing.

## Motivation

Deepfake videos are becoming increasingly realistic and pose a threat to information authenticity. Automated detection methods are necessary to identify manipulated content. GAN-based detection models are effective because adversarial training helps the model learn subtle artifacts introduced during synthesis or face-swapping.

## Methodology

1. **Frame Extraction & Face Cropping**

   * Videos are sampled at configurable frame rates.
   * Faces are detected and aligned using MTCNN or Dlib.
   * Cropped faces are resized to a fixed input size (e.g., 128×128 or 224×224).

2. **Preprocessing**

   * Pixel normalization, optional histogram equalization, and data augmentation (flip, rotation, color jitter).
   * Optional computation of optical flow or temporal differences for motion artifacts.

3. **GAN-based Classifier**

   * `Generator`: produces perturbed faces during training.
   * `Discriminator` / `Classifier`: convolutional network that outputs a probability of `real` vs `fake`.
   * Adversarial training ensures the discriminator learns robust features.

4. **Training Losses**

   * Binary cross-entropy for classification.
   * Adversarial loss between generator and discriminator.
   * Optional feature-matching or perceptual loss for stability.

5. **Inference**

   * Extract face crops from video frames.
   * Run classifier to get per-frame scores.
   * Aggregate scores (mean, median, or temporal model) to produce final video-level prediction.

## Dataset

Supported datasets:

* **DFDC (Deepfake Detection Challenge)**

> Datasets are not included. Download separately and place in the `data/` folder.

## Installation

```bash
git clone https://github.com/MegavathAnil/deepfake-video-detection-using-GAN.git
cd deepfake-video-detection-using-GAN
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Dependencies: PyTorch, torchvision, OpenCV, MTCNN/Dlib, NumPy, tqdm, Streamlit.

## Usage

### Prepare Dataset

```bash
python scripts/extract_faces.py --input-dir data/videos --output-dir data/faces --fps 2 --detector mtcnn
```

### Train Model

```bash
python train.py --data-dir data/faces --epochs 40 --batch-size 32 --lr 1e-4 --save-dir checkpoints/
```

### Evaluate Model

```bash
python evaluate.py --checkpoint checkpoints/model_final.pth --data-dir data/faces --metrics auc accuracy
```

### Inference on Single Video

```bash
python infer.py --video sample_videos/input.mp4 --checkpoint checkpoints/model_final.pth --output results/output.json
```

### Run Streamlit Demo

```bash
streamlit run app.py
```

Upload a video in the demo to see predictions.

## Project Structure

```
deepfake-video-detection-using-GAN/
├── README.md
├── requirements.txt
├── app.py                # Streamlit demo
├── train.py
├── evaluate.py
├── infer.py
├── backend/
│   ├── model.py
│   ├── face_utils.py
│   └── predictor.py
├── scripts/
│   └── extract_faces.py
├── checkpoints/
├── data/
└── assets/
```

## Evaluation Metrics

* Accuracy
* AUC (ROC)
* Precision, Recall, F1-score
* Optional: Equal Error Rate (EER)

## Contribution

Open to contributions! Please submit issues or PRs with clear descriptions.

## License

MIT License.

## Acknowledgements

* FaceForensics++, DFDC datasets
* PyTorch, OpenCV, MTCNN libraries
* Related research papers in deepfake detection

## Contact

**Maintainer:** Anil Megavath
**Email:** [anilkumarmegavath26@gmail.com](anilkumarmegavath26@gmail.com)
**GitHub:** [https://github.com/MegavathAnil](https://github.com/MegavathAnil)
