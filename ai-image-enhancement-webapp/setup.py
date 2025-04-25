import os
import subprocess
import urllib.request

def clone_repo(repo_url, folder_name):
    if not os.path.exists(folder_name):
        print(f"🔄 Cloning {repo_url}...")
        subprocess.run(['git', 'clone', repo_url, folder_name], check=True)
    else:
        print(f"✅ {folder_name} already exists, skipping clone.")

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"⬇️ Downloading {url}...")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        print(f"✅ Downloaded to {dest_path}")
    else:
        print(f"✅ {dest_path} already exists, skipping download.")

def main():
    print("🚀 Setting up model repositories and pretrained weights...")

    # Clone repos
    clone_repo("https://github.com/cszn/BSRGAN.git", "BSRGAN")
    clone_repo("https://github.com/xinntao/Real-ESRGAN.git", "Real-ESRGAN")
    clone_repo("https://github.com/JingyunLiang/SwinIR.git", "SwinIR")

    # Download pretrained models
    download_file(
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth",
        "BSRGAN/model_zoo/BSRGAN.pth"
    )
    download_file(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth"
    )
    download_file(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "SwinIR/model_zoo/SwinIR-L_x4_GAN.pth"
    )

    print("🎉 All models and weights are ready!")

if __name__ == "__main__":
    main()
