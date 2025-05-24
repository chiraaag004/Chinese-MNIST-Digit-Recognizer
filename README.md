# Chinese Digit Recognizer (0–14)

This is a Streamlit-based web app that allows users to draw or upload images of handwritten **Chinese digits (0–14)** and predicts the digit using pre-trained neural network models.

---

## Features

- Draw digits directly on a canvas
- Upload handwritten digit images (JPG/PNG)
- Choose from three trained models:
  - Gradient Descent
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
- Preprocesses images with:
  - Grayscale conversion
  - Inversion (if needed)
  - Auto-cropping to the digit's bounding box
  - Resizing to 64×64 input (model-ready)

---

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- [OpenCV](https://opencv.org/) for image processing
- Pre-trained models saved as `.pkl` files

---

## Installation

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
