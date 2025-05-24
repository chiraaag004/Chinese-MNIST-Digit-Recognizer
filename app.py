import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import pickle



# === Load all model weights ===
with open("model_grad.pkl", "rb") as f:
    model_grad = pickle.load(f)
W1_grad, b1_grad = model_grad['W1'], model_grad['b1']
W2_grad, b2_grad = model_grad['W2'], model_grad['b2']
W3_grad, b3_grad = model_grad['W3'], model_grad['b3']

with open("model_batch.pkl", "rb") as f:
    model_batch = pickle.load(f)
W1_batch, b1_batch = model_batch['W1'], model_batch['b1']
W2_batch, b2_batch = model_batch['W2'], model_batch['b2']
W3_batch, b3_batch = model_batch['W3'], model_batch['b3']

with open("model_sgd.pkl", "rb") as f:
    model_sgd = pickle.load(f)
W1_sgd, b1_sgd = model_sgd['W1'], model_sgd['b1']
W2_sgd, b2_sgd = model_sgd['W2'], model_sgd['b2']
W3_sgd, b3_sgd = model_sgd['W3'], model_sgd['b3']




# === Define forward pass ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def predict_digit(img_array, W1, b1, W2, b2, W3, b3):
    x = img_array.reshape(-1, 1) / 255.0  # Flatten and normalize

    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)

    z3 = np.dot(W3, a2) + b3
    a3 = softmax(z3)

    return np.argmax(a3)




# === Streamlit App ===
st.title("Chinese Digit Recognizer")
st.markdown("Draw a **Chinese digit** (0–14) and get the predicted label!")

# Select model dropdown
model_choice = st.selectbox(
    "Select Model for Prediction",
    ("Gradient Descent Model", "Batch Gradient Model", "SGD Model")
)

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Convert to grayscale image
    img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"), mode="RGB").convert("L")
    img = img.resize((64, 64))  # Match model input size
    img_array = 255 - np.array(img)

    st.image(img_array, caption="Input to Model", width=150, clamp=True)

    # Select weights based on model choice
    if model_choice == "Gradient Descent Model":
        prediction = predict_digit(img_array, W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad)
    elif model_choice == "Batch Gradient Model":
        prediction = predict_digit(img_array, W1_batch, b1_batch, W2_batch, b2_batch, W3_batch, b3_batch)
    else:  # SGD Model
        prediction = predict_digit(img_array, W1_sgd, b1_sgd, W2_sgd, b2_sgd, W3_sgd, b3_sgd)

    st.success(f"### Predicted Digit: {prediction}")