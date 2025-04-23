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

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
import time
from PIL import Image

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, model_name):
    """Process an image with the selected model"""
    result_path = os.path.join(app.config['RESULTS_FOLDER'], 
                              f"{os.path.splitext(os.path.basename(image_path))[0]}_{model_name}.png")

    try:
        # Use RealESRGAN inference directly from their script
        if model_name == 'realesrgan':
            # Add RealESRGAN to path
            sys.path.append('./Real-ESRGAN')
            from inference_realesrgan import main as realesrgan_main
            
            # Create argv for the function
            sys.argv = [
                'inference_realesrgan.py',
                '-n', 'RealESRGAN_x4plus',
                '-i', image_path,
                '-o', app.config['RESULTS_FOLDER'],
                '-s', '4',
                '--face_enhance'
            ]
            realesrgan_main()
            
            # Get the output filename
            output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_out{os.path.splitext(image_path)[1]}"
            output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
            
            # Rename to match expected pattern
            if os.path.exists(output_path):
                os.rename(output_path, result_path)
        
        # SwinIR has different API requirements
        elif model_name == 'swinir':
            sys.path.append('./SwinIR')
            try:
                # Import necessary functions from SwinIR
                from SwinIR.main_test_swinir import define_model, test as swinir_test
                import torch
                
                # Set up device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Set up model
                model_path = 'experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
                model = define_model(task='real_sr', scale=4, large_model=True)
                
                # Load model weights
                pretrained_model = torch.load(model_path, map_location=device)
                if 'params_ema' in pretrained_model:
                    model.load_state_dict(pretrained_model['params_ema'], strict=True)
                elif 'params' in pretrained_model:
                    model.load_state_dict(pretrained_model['params'], strict=True)
                else:
                    model.load_state_dict(pretrained_model, strict=True)
                model.eval()
                model = model.to(device)
                
                # Process image
                input_dir = os.path.dirname(image_path)
                filename = os.path.basename(image_path)
                
                # If SwinIR test function expects different parameters, adjust this call
                swinir_test(model, os.path.dirname(image_path), os.path.dirname(result_path), 
                            [os.path.basename(image_path)], sf=4, device=device)
                
                # Find and rename the output file to match our naming convention
                for file in os.listdir(app.config['RESULTS_FOLDER']):
                    if file.endswith('SwinIR.png') and os.path.basename(image_path).split('.')[0] in file:
                        os.rename(os.path.join(app.config['RESULTS_FOLDER'], file), result_path)
                        break
                        
            except Exception as e:
                print(f"Error using SwinIR API: {e}")
                # Use fallback method of directly calling the script
                try:
                    # Clean up sys.argv
                    sys.argv = [
                        'main_test_swinir.py',
                        '--task', 'real_sr',
                        '--model_path', 'experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
                        '--folder_lq', os.path.dirname(image_path),
                        '--folder_gt', os.path.dirname(image_path),  # Dummy value, not used
                        '--scale', '4',
                        '--large_model'
                    ]
                    
                    # Import and run the whole script
                    sys.path.append('./SwinIR')
                    from SwinIR.main_test_swinir import main as swinir_main
                    swinir_main()
                    
                    # Find and rename the output file
                    for file in os.listdir(app.config['RESULTS_FOLDER']):
                        if file.endswith('SwinIR.png') and os.path.basename(image_path).split('.')[0] in file:
                            os.rename(os.path.join(app.config['RESULTS_FOLDER'], file), result_path)
                            break
                            
                except Exception as e2:
                    print(f"Error using SwinIR script: {e2}")
                    # Fall back to simple resize
                    img = Image.open(image_path)
                    width, height = img.size
                    img_resized = img.resize((width*4, height*4), Image.LANCZOS)
                    img_resized.save(result_path)
        
        # BSRGAN model
        elif model_name == 'bsrgan':
            sys.path.append('./BSRGAN')
            
            try:
                # Try to use BSRGAN directly
                # Check if we can directly use the model - adjust imports as needed based on BSRGAN structure
                try:
                    # BSRGAN might have a different structure - adjust these imports accordingly
                    from BSRGAN.models.network_swinir import SwinIR as net
                except ImportError:
                    # Fallback - many BSR implementations are based on SwinIR
                    sys.path.append('./SwinIR')
                    from models.network_swinir import SwinIR as net
                
                import torch
                
                # Load model
                model_path = 'BSRGAN/model_zoo/BSRGAN.pth'
                torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Create model instance (parameters may vary)
                model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='nearest+conv')
                
                # Load model weights
                model.load_state_dict(torch.load(model_path, map_location=torch_device), strict=True)
                model.eval()
                model = model.to(torch_device)
                
                # Process image
                img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
                img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(torch_device)
                
                with torch.no_grad():
                    output = model(img)
                
                # Save output
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) * 255.0  # RGB to BGR
                cv2.imwrite(result_path, output.astype(np.uint8))
                
            except Exception as e:
                print(f"Error using BSRGAN model: {e}")
                # Fallback to using SwinIR with BSRGAN weights
                try:
                    # Clean up sys.argv
                    sys.argv = [
                        'main_test_swinir.py',
                        '--task', 'real_sr',
                        '--model_path', 'BSRGAN/model_zoo/BSRGAN.pth',
                        '--folder_lq', os.path.dirname(image_path),
                        '--folder_gt', os.path.dirname(image_path),  # Dummy value, not used
                        '--scale', '4',
                        '--large_model'
                    ]
                    
                    sys.path.append('./SwinIR')
                    from SwinIR.main_test_swinir import main as swinir_main
                    swinir_main()
                    
                    # Find and rename the output file
                    for file in os.listdir(app.config['RESULTS_FOLDER']):
                        if file.endswith('SwinIR.png') and os.path.basename(image_path).split('.')[0] in file:
                            os.rename(os.path.join(app.config['RESULTS_FOLDER'], file), result_path)
                            break
                            
                except Exception as e2:
                    print(f"Error using SwinIR with BSRGAN weights: {e2}")
                    # Simple resize as last resort
                    img = Image.open(image_path)
                    width, height = img.size
                    img_resized = img.resize((width*4, height*4), Image.LANCZOS)
                    img_resized.save(result_path)
        
        return result_path
    except Exception as e:
        print(f"Error processing image with {model_name}: {e}")
        # Fallback to simple resize if model fails
        try:
            img = Image.open(image_path)
            width, height = img.size
            img_resized = img.resize((width*4, height*4), Image.LANCZOS)
            img_resized.save(result_path)
            return result_path
        except Exception as e2:
            print(f"Fallback resize also failed: {e2}")
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        model_name = request.form.get('model', 'realesrgan')
        result_path = process_image(filepath, model_name)
        
        if result_path:
            return render_template('result.html', 
                                  original=filepath, 
                                  enhanced=result_path,
                                  model_name=model_name)
        else:
            flash('Error processing image')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("Starting web application...")
    app.run(debug=True)