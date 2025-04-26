import os
import uuid
import shutil
import streamlit as st
from inference_utils import run_model
from setup import main as setup_models

# ---------- Helper Functions ----------

def list_files_in_folder(folder_path):
    """List all files inside a given folder."""
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

# ---------- Initial Setup ----------

# Create upload/results folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Clone/download models if missing
if not (os.path.exists('BSRGAN') and os.path.exists('Real-ESRGAN') and os.path.exists('SwinIR')):
    st.info("🚀 Setting up models and pretrained weights...")
    setup_models()

# ---------- Streamlit App UI ----------

st.title("🖼️ AI Image Enhancement App")

# Sidebar for file explorer
st.sidebar.title("🗂️ Explore Server Files")
folder = st.sidebar.selectbox("Select a folder to view files", [
    "BSRGAN/model_zoo",
    "Real-ESRGAN/experiments/pretrained_models",
    "SwinIR/model_zoo",
    "static/uploads",
    "static/results"
])

if folder:
    if os.path.exists(folder):
        files = list_files_in_folder(folder)
        with st.sidebar.expander(f"📂 Files inside `{folder}` ({len(files)})", expanded=True):
            if files:
                for f in files:
                    size = os.path.getsize(f) / (1024*1024)
                    st.write(f"📄 {os.path.basename(f)} — {size:.2f} MB")
            else:
                st.warning("⚠️ No files found.")
    else:
        st.sidebar.error(f"❌ Folder `{folder}` not found.")

# Main Area
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("🧠 Choose a model", ["bsrgan", "realesrgan", "swinir"])

if uploaded_file and model_choice:
    if st.button("✨ Enhance Image"):
        # Save uploaded file
        name, ext = os.path.splitext(uploaded_file.name)
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        filename = f"{safe_name}_{str(uuid.uuid4())[:8]}{ext}"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(RESULT_FOLDER, filename)
        
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Run selected model
        success = run_model(input_path, output_path, model_choice)

        # Display result
        if success and os.path.exists(output_path):
            st.success("✅ Image enhanced successfully!")
            st.image(output_path, caption="Enhanced Image", use_column_width=True)
            with open(output_path, "rb") as f:
                st.download_button(label="⬇️ Download Enhanced Image", data=f, file_name=filename)
        else:
            st.error("❌ Enhancement failed. Please try again.")
