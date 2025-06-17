# Semantic Segmentation of Aerial Images Using U-Net

This project applies semantic segmentation to high-resolution satellite images of Dubai using deep learning, specifically the U-Net architecture. The objective is to classify each pixel into one of several land cover types such as buildings, roads, vegetation, water, and more.

---

## 📊 Dataset

- **Source:** [Kaggle - Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery)
- **Description:** Aerial images of Dubai with pixel-wise semantic segmentation.
- **Classes:**
  - Building (#3C1098)
  - Land / Unpaved (#8429F6)
  - Road (#6EC1E4)
  - Vegetation (#FEDD3A)
  - Water (#E2A929)
  - Unlabeled (#9B9B9B)
- **Preprocessing:**
  - Cropped to size divisible by 256
  - Converted to 256×256 patches using `patchify`
  - RGB masks converted to categorical labels

---

## 🧠 Model Architecture

Two models are trained and compared:
1. **Standard U-Net**: A custom-built U-Net model
2. **Pretrained U-Net**: U-Net with a ResNet34 backbone from `segmentation_models` library

### Loss & Metrics:
- Loss: `categorical_crossentropy` or `Dice Loss + Focal Loss`
- Metrics: `Accuracy` and `Jaccard Coefficient (IoU)`

---

## 🚀 Training

- Data split: 80% Training, 20% Testing
- Batch size: 16
- Epochs: 20 (adjustable)
- Early stopping and model checkpoint used for best performance

---

## 🖼️ Visualization

The notebook includes visualization of:
- Input satellite images
- Ground truth masks
- Predicted segmentation masks

---

## 📈 Evaluation

- IoU scores computed using both custom metric and Keras's `MeanIoU`
- Achieved validation IoU of ~0.75 using `categorical_crossentropy`

---

## 🧪 Requirements

- Python 3.7+
- TensorFlow / Keras
- segmentation_models
- OpenCV, NumPy, Matplotlib, Patchify, PIL
- Scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt

