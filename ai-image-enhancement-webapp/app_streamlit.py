import os
import uuid
import shutil
import streamlit as st
from inference_utils import run_model
from setup import main as setup_models

# ----------------------------- #
# 🛠️ Setup Check Functions
# ----------------------------- #

def check_model_setup():
    status = {}

    # Check folders
    status['BSRGAN'] = os.path.exists('BSRGAN')
    status['Real-ESRGAN'] = os.path.exists('Real-ESRGAN')
    status['SwinIR'] = os.path.exists('SwinIR')

    # Check pre-trained model files
    status['BSRGAN_model'] = os.path.exists('BSRGAN/model_zoo/BSRGAN.pth')
    status['RealESRGAN_model'] = os.path.exists('Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth')
    status['SwinIR_model'] = os.path.exists('SwinIR/model_zoo/SwinIR-L_x4_GAN.pth')

    return status

# ----------------------------- #
# 🖥️ Streamlit UI
# ----------------------------- #

st.set_page_config(page_title="Image Enhancement App", page_icon="✨", layout="wide")

st.title("🖼️ Image Enhancement App")

# Status Check
st.subheader("🔍 Checking Model Setup...")

status = check_model_setup()

# Display folder status
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📦 BSRGAN: {'✅' if status['BSRGAN'] else '❌'}")
with col2:
    st.info(f"📦 Real-ESRGAN: {'✅' if status['Real-ESRGAN'] else '❌'}")
with col3:
    st.info(f"📦 SwinIR: {'✅' if status['SwinIR'] else '❌'}")

# Display model weights status
col4, col5, col6 = st.columns(3)
with col4:
    st.success(f"🎯 BSRGAN.pth: {'✅' if status['BSRGAN_model'] else '❌'}")
with col5:
    st.success(f"🎯 RealESRGAN.pth: {'✅' if status['RealESRGAN_model'] else '❌'}")
with col6:
    st.success(f"🎯 SwinIR-L_x4_GAN.pth: {'✅' if status['SwinIR_model'] else '❌'}")

# Auto run setup if missing
if not all(status.values()):
    st.warning("⚙️ Some models or weights are missing. Setting up now...")
    setup_models()
    st.success("✅ Setup complete! Ready to go!")

st.markdown("---")

# ----------------------------- #
# 🖼️ Image Upload & Enhancement
# ----------------------------- #

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("🛠️ Choose a model", ["bsrgan", "realesrgan", "swinir"])

if uploaded_file and model_choice:
    if st.button("✨ Enhance Image"):
        # Save uploaded image
        name, ext = os.path.splitext(uploaded_file.name)
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        filename = f"{safe_name}_{str(uuid.uuid4())[:8]}{ext}"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(RESULT_FOLDER, filename)

        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Run enhancement
        with st.spinner(f"🖼️ Enhancing image using {model_choice.upper()}..."):
            success = run_model(input_path, output_path, model_choice)

        if success and os.path.exists(output_path):
            st.success("🎉 Enhancement Complete!")
            st.image(output_path, caption="Enhanced Image", use_column_width=True)
            with open(output_path, "rb") as f:
                st.download_button(label="📥 Download Enhanced Image", data=f, file_name=filename)
        else:
            st.error("❌ Enhancement failed. Please check the logs and try again.")
