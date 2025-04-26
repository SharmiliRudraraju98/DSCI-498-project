import os
import uuid
import shutil
import streamlit as st
from inference_utils import run_model
from setup import main as setup_models

# Initialize setup
if not (os.path.exists('BSRGAN') and os.path.exists('Real-ESRGAN') and os.path.exists('SwinIR')):
    st.info("Setting up models...")
    setup_models()

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

st.title("🖼 AI Image Enhancement App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose a model", ["bsrgan", "realesrgan", "swinir"])

if uploaded_file and model_choice:
    if st.button("Enhance Image"):
        # Save uploaded image
        name, ext = os.path.splitext(uploaded_file.name)
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        filename = f"{safe_name}_{str(uuid.uuid4())[:8]}{ext}"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(RESULT_FOLDER, filename)
        
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Run model
        success = run_model(input_path, output_path, model_choice)
        
        if success and os.path.exists(output_path):
            st.success("✨ Image enhanced successfully!")
            st.image(output_path, caption="Enhanced Image", use_column_width=True)
            with open(output_path, "rb") as f:
                st.download_button(label="Download Enhanced Image", data=f, file_name=filename)
        else:
            st.error("Enhancement failed. Please try again.")
