# 🖼️ Image Restoration Using Real-ESRGAN, SwinIR, and BSRGAN

## 📚 Project Description

This project focuses on enhancing the quality of degraded images using three state-of-the-art deep learning models: Real-ESRGAN, SwinIR-Large, and BSRGAN.

We implemented a full evaluation pipeline in a Jupyter Notebook, followed by building a Flask web application for user interaction.

The project compares the models based on:
- Quantitative metrics (PSNR and SSIM)
- Visual comparisons through zoomed inspection
- Color histogram and brightness distribution analysis
- Edge detection and preservation measurements

## 🛠️ How to Run the Jupyter Notebook

1. **Clone the repository or download the project zip:**
   ```bash
   git clone https://github.com/SharmiliRudraraju98/DSCI-498-project
   ```

2. **Open the provided Jupyter Notebook:**
   - Navigate to the `Restoring_Image_Quality_With_AI_using_Real_ESRGAN,_SwinIR_and_BSRGAN_SharmiliR (3).ipynb`
3. **Install the required Python packages:**
   - torch
   - opencv-python
   - matplotlib
   - scikit-image
   - shutil
   - glob
   - facexlib
   - basicsr
   - gfpgan
   - timm

4. **Run the Notebook Cells Sequentially:**
   - Clone model repositories
   - Download pretrained models
   - Upload your images
   - Run inference for all three models (Real-ESRGAN, BSRGAN, and SwinIR-Large)
   - View enhanced results and evaluation metrics

5. **Results:**
   - Enhanced images are saved under the `results/` directory
   - Evaluation CSVs (`full_image_restoration_metrics.csv`, `average_metrics_per_model.csv`) are generated automatically

## 📋 Add on Setup Instructions for local webapp

###Clone the repository or download the project zip:**
   ```bash
   git clone https://github.com/SharmiliRudraraju98/DSCI-498-project
   ```

### Clone Model Repositories /DSCI-498-project/ai-image-enhancement-webapp
```bash
git clone https://github.com/cszn/BSRGAN.git
git clone https://github.com/xinntao/Real-ESRGAN.git
git clone https://github.com/JingyunLiang/SwinIR.git
```

### Create Required Directories
```bash
mkdir -p BSRGAN/model_zoo
mkdir -p Real-ESRGAN/experiments/pretrained_models
mkdir -p SwinIR/model_zoo
```

### Download Pre-trained Models
```bash
# BSRGAN model
curl -L -o BSRGAN/model_zoo/BSRGAN.pth https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth

# Real-ESRGAN model
curl -L -o Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# SwinIR model
curl -L -o SwinIR/model_zoo/SwinIR-L_x4_GAN.pth https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
```

### Install Real-ESRGAN
```bash
cd Real-ESRGAN
pip install -e . --user
```

### CPU/GPU Device Selection in BSRGAN/main_test_bsrgan.py
```python
if torch.cuda.is_available():
    device = torch.device('cuda')
    logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
else:
    device = torch.device('cpu')
    logger.info('{:>16s} : CPU mode'.format('Device'))
if 'img_gt' in locals() and img_gt is not None:
    # Further processing code
```

### install requirements.txt
```bash
pip install -r requirements.txt
'
```

### Run app.py
```bash
python app.py
'
```
## 🌐 Web Application

The project includes a Flask web application that provides an interactive interface for users to upload images and apply different restoration models. The web app demonstrates real-world application of these models and allows for easy comparison of results.

To run the web application:

1. Navigate to the web app directory
2. Install Flask and other required dependencies
3. Run the Flask application
4. Access the web interface through your browser at the specified address

For more details on the web application setup and usage, see the web application documentation in the project repository.
