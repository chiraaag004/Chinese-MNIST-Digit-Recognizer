# 🧠 Chinese Digit Recognizer (0–14)

This is a Streamlit-based web app that allows users to draw or upload images of handwritten **Chinese digits (0–14)** and predicts the digit using pre-trained neural network models.

🌐 **Live Demo**: [Open the App](https://chinese-mnist-digit-recognizer-qjbmzzqlcpuzufdjsjpxq8.streamlit.app/)

---

## 📚 About the Neural Network

This app uses a simple **feedforward neural network built from scratch** using NumPy. No deep learning frameworks (like TensorFlow or PyTorch) were used.

- Input size: 64 × 64 grayscale image (flattened)
- Hidden layers: 2 layers with ReLU activation
- Output layer: Softmax layer for 15-class classification
- Trained using:
  - Gradient Descent
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)

---

## 📦 Dataset

The app is trained on the **Chinese-MNIST** dataset:
- 15,000 images of handwritten Chinese digits (0–14)
- Each image is 64×64 grayscale

📂 Dataset Link: [Chinese-MNIST on Kaggle](https://www.kaggle.com/datasets/gpreda/chinese-mnist)

---

## 🚀 Features

- ✏️ Draw digits directly on a canvas
- 📤 Upload handwritten digit images (JPG/PNG)
- 🔀 Choose from three trained models:
  - Gradient Descent
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
- 🖼️ Preprocessing:
  - Grayscale conversion
  - Inversion (if needed)
  - Auto-cropping to digit's bounding box
  - Resizing to 64×64 input (model-ready)

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- [OpenCV](https://opencv.org/) (via `opencv-python-headless`)
- Custom neural network built without deep learning libraries

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/chiraaag004/Chinese-MNIST-Digit-Recognizer.git
cd Chinese-MNIST-Digit-Recognizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```
