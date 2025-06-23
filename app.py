import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import pandas as pd
import altair as alt

from convolutional import Convolutional
from dense import Dense
from reshape import Reshape
from activation_layers import Softmax, ReLU


st.set_page_config(page_title="Chinese Digit Recognizer", layout="wide")

# === Digit labels with Chinese symbols ===
digit_map = {
    0: "0 零", 1: "1 一", 2: "2 二", 3: "3 三", 4: "4 四", 5: "5 五",
    6: "6 六", 7: "7 七", 8: "8 八", 9: "9 九", 10: "10 十",
    11: "100 百", 12: "1000 千", 13: "10000 万", 14: "100000000 亿"
}

# === Utility: Load weights from .npz ===
def load_weights_auto(network, filename="network_weights.npz"):
    data = np.load(filename)
    for i, layer in enumerate(network):
        for attr in ["weights", "bias", "kernels", "biases"]:
            key = f"layer_{i}_{attr}"
            if key in data:
                setattr(layer, attr, data[key])

# === Define the network ===
@st.cache_resource
def get_model():
    network = [
        Convolutional((1, 64, 64), 5, 8),
        ReLU(),
        Reshape((8, 60, 60), (8 * 60 * 60,)),
        Dense(8 * 60 * 60, 256),
        ReLU(),
        Dense(256, 15),
        Softmax()
    ]
    weight_path = "D:\chirag\chinese_digit_recogniser\model_weights.npz"
    if not os.path.exists(weight_path):
        st.error(f"Model weights not found at {weight_path}")
        st.stop()
    load_weights_auto(network, weight_path)
    return network

network = get_model()

# === Prediction utilities ===
def predict(network, x):
    output = x
    for layer in network:
        output = layer.forward(output)
    return output

# === Streamlit UI setup ===

st.markdown(
    """
    <style>
    .stCanvas > div {border-radius: 12px; border: 1px solid #ccc;}
    .prediction-box {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        border: 2px solid #1f77b4;
        border-radius: 12px;
        padding: 0.3rem;
        margin-bottom: 1rem;
        background-color: #f7fafd;
    }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖌️ Chinese Digit Recognizer")

col1, col2 = st.columns([1, 2.5], gap="medium")

with col1:
    st.subheader("Draw a Digit")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=4,
        stroke_color="black",
        background_color="white",
        height=220,
        width=220,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

with col2:
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, :3].astype("uint8"), mode="RGB").convert("L")
        img = img.resize((64, 64))
        img_array = (255.0 - np.array(img))  # Invert: white background to black stroke

        # Normalize and reshape to match input format: (1, 1, 64, 64)
        x_input = img_array.reshape(1, 1, 64, 64)

        probs = predict(network, x_input).ravel()
        pred_idx = int(np.argmax(probs))
        pred_label = digit_map[pred_idx]

        st.markdown(f'<div class="prediction-box">{pred_label}</div>', unsafe_allow_html=True)

        df_chart = pd.DataFrame({
            "Digit": [digit_map[i] for i in range(15)],
            "Confidence": probs
        })

        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X('Digit', sort=list(digit_map.values()), title="Digit"),
            y=alt.Y('Confidence', title="Softmax Confidence", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Digit", alt.Tooltip("Confidence", format=".2f")]
        ).properties(
            height=300,
            width=350
        ).configure_axisX(labelAngle=1)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Draw a digit on the canvas to see prediction.")
