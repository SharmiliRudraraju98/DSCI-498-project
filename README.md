# AI-Powered Image Restoration System

## Project Overview
This project explores state-of-the-art AI models for enhancing and restoring degraded images. I'm implementing and comparing three deep learning approaches (Real-ESRGAN, SwinIR, and BSRGAN) to address various image quality issues while developing a user-friendly interface and systematic evaluation framework.

## Problem Statement
Image degradation is a common challenge across photography, digital archives, and media. Issues include:
- Low resolution and loss of detail
- Noise and visual artifacts
- Blur and lack of sharpness
- Compression artifacts
- Color degradation

Traditional methods often struggle with these complex issues, especially when multiple degradation types co-exist in a single image. This project leverages recent advancements in AI to overcome these limitations.

## Data Information
The project uses:
- **RealSRSet** - A benchmark dataset proposed in the BSRGAN paper (ICCV 2021) containing real-world low-quality images
- **Custom test dataset** - I'm creating a structured collection of test images across several categories:
  - Portrait/facial images
  - Landscape and nature scenes
  - Urban/architectural photographs
  - Text documents
  - Images with fine textures and details
  - Low-light photography
  - Images with compression artifacts

This diverse dataset will enable systematic evaluation of each model's performance across different image types and degradation scenarios.

## Project Goals

### 1. Implementation of AI Models
- Set up Real-ESRGAN, SwinIR, and BSRGAN in a Google Colab environment
- Create a user-friendly interface for easy access to these powerful models
- Ensure efficient processing with options for memory management

### 2. Systematic Evaluation Framework
- Develop quantitative metrics (PSNR, SSIM, LPIPS) for objective comparison
- Create a structured testing methodology across different image categories
- Identify specific strengths and weaknesses of each model

### 3. Enhanced User Experience
- Design an intuitive interface with clear explanations
- Implement interactive model selection and parameter adjustment
- Create visualization tools for detailed result comparison
- Add export functionality for processed images

### 4. Documentation and Analysis
- Provide comprehensive documentation on model architectures and capabilities
- Analyze performance patterns across different image types
- Create visual representations of comparative results

## Current Progress
- ✅ Research and selection of AI models
- ✅ Setting up the base implementation environment
- 🔄 Initial code implementation for all three models
- 🔄 Basic visualization of results
- 🔄 Building the structured test dataset
- 🔄 Developing the evaluation framework

## Future Work (Milestone 2)
- Complete the systematic evaluation across all image categories
- Implement the enhanced user interface
- Add detailed metrics and analysis dashboard
- Optimize performance for large images

## References
- Real-ESRGAN: [GitHub](https://github.com/xinntao/Real-ESRGAN) | [Paper](https://doi.org/10.48550/arXiv.2107.10833)
- SwinIR: [GitHub](https://github.com/JingyunLiang/SwinIR) | [Paper](https://doi.org/10.48550/arXiv.2108.10257)
- BSRGAN: [GitHub](https://github.com/cszn/BSRGAN) | [Paper](https://doi.org/10.48550/arXiv.2103.14006)

## Implementation Details
The project is implemented in Python using PyTorch, with Google Colab providing GPU acceleration. The implementation leverages pre-trained models while adding custom evaluation metrics and an enhanced user interface.
