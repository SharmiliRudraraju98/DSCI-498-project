# 🖼️ Image Restoration Using Real-ESRGAN, SwinIR, and BSRGAN

## 📚 Project Description

This project focuses on enhancing the quality of degraded images using three state-of-the-art deep learning models: Real-ESRGAN, SwinIR-Large, and BSRGAN.

We implemented a full evaluation pipeline in a Jupyter Notebook, followed by building a Flask web application(add-on/optional) for user interaction.

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
   - Navigate to the `notebooks/` directory and open `image_restoration_colab.ipynb`.

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
   - Evaluation CSVs (`full_image_restoration_metrics.csv`, `average_metrics_per_model.csv`) are generated automatically.
  
   - 
