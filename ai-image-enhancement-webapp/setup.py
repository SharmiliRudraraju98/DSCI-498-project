# setup_windows.py
import os
import subprocess
import sys
import urllib.request
import zipfile
import shutil

def run_command(command):
    """Run a command and print its output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())
        
    return process.returncode

def download_file(url, destination):
    """Download a file using urllib instead of wget"""
    print(f"Downloading {url} to {destination}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded successfully to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def setup_environment():
    """Set up the environment for the web application"""
    print("Setting up environment for AI Image Enhancement Web App...")
    
    # Create directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('experiments/pretrained_models', exist_ok=True)
    os.makedirs('BSRGAN/model_zoo', exist_ok=True)
    os.makedirs('Real-ESRGAN/experiments/pretrained_models', exist_ok=True)
    
    # Install dependencies using python -m pip to avoid path issues
    print("Installing dependencies...")
    dependencies = [
        "flask",
        "torch torchvision --index-url https://download.pytorch.org/whl/cu121",  # CUDA 12.1 version
        "opencv-python",
        "numpy",
        "pillow",
        "matplotlib",
        "timm"
    ]
    
    for dep in dependencies:
        run_command(f"python -m pip install {dep}")
    
    # Handle special packages that might cause issues
    run_command("python -m pip install basicsr")
    run_command("python -m pip install facexlib")
    run_command("python -m pip install gfpgan")
    
    # Clone repositories
    print("Cloning model repositories...")
    repositories = [
        "https://github.com/xinntao/Real-ESRGAN.git",
        "https://github.com/cszn/BSRGAN.git",
        "https://github.com/JingyunLiang/SwinIR.git"
    ]
    
    for repo in repositories:
        repo_name = repo.split('/')[-1].split('.')[0]
        if not os.path.exists(repo_name):
            run_command(f"git clone {repo}")
    
    # Install Real-ESRGAN
    print("Setting up Real-ESRGAN...")
    os.chdir('Real-ESRGAN')
    run_command("python setup.py develop")
    os.chdir('..')
    
    # Download pre-trained models
    print("Downloading pre-trained models...")
    model_urls = [
        ("https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth", "BSRGAN/model_zoo/BSRGAN.pth"),
        ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth"),
        ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth", "experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
    ]
    
    for url, destination in model_urls:
        download_file(url, destination)
    
    # Fix import issues for torchvision
    print("Creating compatibility patch for basicsr...")
    
    # Create patch file for basicsr
    patch_dir = "patches"
    os.makedirs(patch_dir, exist_ok=True)
    
    with open(os.path.join(patch_dir, "functional_tensor.py"), "w") as f:
        f.write("""
# Compatibility patch for basicsr
def rgb_to_grayscale(img):
    import torch
    import torch.nn.functional as F
    
    # Implementation similar to the original but compatible with newer torchvision
    r, g, b = img.unbind(dim=-3)
    # This uses ITU-R BT.709 weights for RGB to grayscale conversion
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).to(img.dtype).unsqueeze(-3)
""")
    
    # Modify app.py to fix import issues
    print("Updating app.py to handle compatibility issues...")
    
    with open("app.py", "r") as f:
        app_content = f.read()
    
    # Add import patch code to the top of app.py 
    import_patch = """
import sys
import os

# Add compatibility patch for torchvision.transforms.functional_tensor
patch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")
sys.path.insert(0, patch_dir)

# Create a mock module to fix the import issue
import sys
class MockModule:
    pass

sys.modules["torchvision.transforms.functional_tensor"] = MockModule()
sys.modules["torchvision.transforms.functional_tensor"].rgb_to_grayscale = __import__("functional_tensor").rgb_to_grayscale
"""
    
    modified_app = import_patch + app_content
    
    with open("app.py", "w") as f:
        f.write(modified_app)
    
    print("Setup complete! You can now run the application with: python app.py")

if __name__ == "__main__":
    setup_environment()