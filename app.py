import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = load_model("mnist_model.h5")  # File needs to be near the script

st.title("numbers guesser")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    x = np.array(img).astype("float32") / 255.0
    x = x.reshape(1, 28, 28, 1)

    pred = model.predict(x)[0]
    st.write("Prediction:", np.argmax(pred))
    st.bar_chart(pred)
