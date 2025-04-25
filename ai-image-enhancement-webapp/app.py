from flask import Flask, render_template, request, send_from_directory
import os
import uuid
from inference_utils import run_model
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image = request.files['image']
    model = request.form['model']

    if not image.filename:
        return "Empty filename", 400

    name, ext = os.path.splitext(image.filename)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)  # sanitize filename
    filename = f"{safe_name}_{str(uuid.uuid4())[:8]}{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(RESULT_FOLDER, filename)

    image.save(input_path)

    success = run_model(input_path, output_path, model)

    if not success or not os.path.exists(output_path):
        return render_template('error.html', model=model, filename=filename)

    return render_template(
        'result.html',
        input_image=input_path,
        output_image=output_path,
        output_filename=filename,
        model=model
    )

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
