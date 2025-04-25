import os
import subprocess
import shutil
import glob

def run_model(input_path, output_path, model_name):
    # Ensure absolute paths
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    # Delete previous output if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # if model_name == 'realesrgan':
    #     # Real-ESRGAN with tile setting for better CPU performance
    #     cmd = [
    #         'python', 'inference_realesrgan.py',
    #         '-n', 'RealESRGAN_x4plus',
    #         '-i', input_path,
    #         '-o', os.path.dirname(output_path),
    #         '--tile', '800',
    #         '--face_enhance'
    #     ]
    #     cwd = 'Real-ESRGAN'
    #     subprocess.run(cmd, cwd=cwd, check=True)
    #     return True

    if model_name == 'realesrgan':
        cmd = [
            'python', 'inference_realesrgan.py',
            '-n', 'RealESRGAN_x4plus',
            '-i', input_path,
            '-o', os.path.dirname(output_path),
            '--tile', '800'
        ]
        cwd = 'Real-ESRGAN'
        try:
            subprocess.run(cmd, cwd=cwd, check=True)
        except subprocess.CalledProcessError as e:
            print("Real-ESRGAN execution failed!")
            print("Stdout:\n", e.stdout)
            print("Stderr:\n", e.stderr)
            return False

        # Explicitly look for output file
        input_filename = os.path.basename(input_path)
        input_name, ext = os.path.splitext(input_filename)
        output_guess = os.path.join(os.path.dirname(output_path), f'{input_name}_out{ext}')
        if os.path.exists(output_guess):
            shutil.copy(output_guess, output_path)
            print(f"Real-ESRGAN output copied to: {output_path}")
            return True
        else:
            print(f"Real-ESRGAN output not found at: {output_guess}")
            return False

    elif model_name == 'bsrgan':
        default_input_folder = os.path.join('BSRGAN', 'testsets', 'RealSRSet')
        default_output_folder = os.path.join('BSRGAN', 'testsets', 'RealSRSet_results_x4')

        # Clean old inputs
        if os.path.exists(default_input_folder):
            shutil.rmtree(default_input_folder)
        os.makedirs(default_input_folder)

        # Clean old outputs
        if os.path.exists(default_output_folder):
            shutil.rmtree(default_output_folder)
        os.makedirs(default_output_folder)

        # Copy uploaded image into expected folder
        input_basename = os.path.basename(input_path)
        input_for_bsrgan = os.path.join(default_input_folder, input_basename)
        shutil.copy(input_path, input_for_bsrgan)

        # Run the BSRGAN model
        cmd = ['python', 'main_test_bsrgan.py']
        cwd = 'BSRGAN'
        subprocess.run(cmd, cwd=cwd, check=True)

        # Find and copy the result
        output_candidates = glob.glob(os.path.join(default_output_folder, '*'))
        matching = [f for f in output_candidates if input_basename.split('.')[0] in os.path.basename(f)]

        if matching:
            enhanced_file = matching[0]
            print(f"Found BSRGAN result: {enhanced_file}")
            shutil.copy(enhanced_file, output_path)
            return True
        else:
            print(f"BSRGAN result not found in: {default_output_folder}")
            print(f"Expected something matching: {input_basename}")
            return False

    elif model_name == 'swinir':
        # Copy image to a temp folder
        temp_input_dir = os.path.join('SwinIR', 'temp_input')
        os.makedirs(temp_input_dir, exist_ok=True)

        temp_image_path = os.path.join(temp_input_dir, os.path.basename(input_path))
        shutil.copy(input_path, temp_image_path)

        cmd = [
            'python', 'main_test_swinir.py',
            '--task', 'real_sr',
            '--scale', '4',
            '--model_path', 'model_zoo/SwinIR-L_x4_GAN.pth',
            '--folder_lq', 'temp_input',
            '--tile', '256',
            '--large_model'
        ]
        cwd = 'SwinIR'

        try:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            print("⚠️ SwinIR stdout:\n", result.stdout)
            print("❌ SwinIR stderr:\n", result.stderr)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            print("💥 SwinIR execution failed!")
            print("📤 Stdout:\n", e.stdout)
            print("📥 Stderr:\n", e.stderr)
            return False

        # Locate result
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join('SwinIR', 'results', 'swinir_real_sr_x4_large')
        output_file_path = os.path.join(output_dir, f"{input_filename}_SwinIR.png")

        if os.path.exists(output_file_path):
            final_output = output_file_path.replace('_SwinIR.png', '_SwinIR_large.png')
            os.rename(output_file_path, final_output)
            shutil.copy(final_output, output_path)
            print(f"✅ SwinIR result copied to {output_path}")
            return True
        else:
            print(f"❌ SwinIR output not found at: {output_file_path}")
            return False    


    else:
        raise ValueError(f"❌ Invalid model name: {model_name}")
