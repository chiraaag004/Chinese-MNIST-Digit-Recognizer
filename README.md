# 🧠 Chinese Digit Recognizer (0–14)

This project focuses on recognizing handwritten **Chinese digits (0–14)** using various neural network models trained on the [Chinese-MNIST dataset](https://www.kaggle.com/datasets/fedesoriano/chinese-mnist-digit-recognizer).

---

## 📚 Model Implementations

The project includes **three different notebooks**, each implementing a distinct approach to compare performance, training complexity, and accuracy:

| Notebook                          | Framework        | Model Type      | Description                                              |
| --------------------------------- | ---------------- | --------------- | -------------------------------------------------------- |
| `Chines_MNIST_NN.ipynb`           | NumPy (Manual)   | Feedforward DNN | Fully connected layers, ReLU activation, Softmax output  |
| `Chines_MNIST_CNN.ipynb`          | NumPy + Custom   | CNN (manual)    | Manual implementation of convolutions, pooling, backprop |
| `Chines_MNIST_tensorflow.ipynb`   | TensorFlow/Keras | CNN (modern)    | High-level API with efficient training and tuning        |

---

## 📦 Dataset

**Chinese-MNIST**:

* 15,000 grayscale images of handwritten digits (0–14)
* Image size: 64 × 64 pixels
* Each digit corresponds to a unique Chinese numeral character

📥 [Download Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/chinese-mnist-digit-recognizer)

---

## 🚀 Features

* 📤 Accepts uploaded images of handwritten Chinese digits
* 🔍 Automatic image preprocessing:

  * Grayscale conversion
  * Inversion correction (if background is black)
  * Auto-cropping to bounding box of digit
  * Resizing to 64×64 input
* 📈 Comparative training and evaluation across all models

---

## 🛠️ Tech Stack

* `NumPy` — matrix ops and manual model building
* `OpenCV` — image processing utilities
* `Pillow (PIL)` — image handling
* `Matplotlib` — training visualization
* `TensorFlow / Keras` — deep learning for fast prototyping
