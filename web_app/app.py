from flask import Flask, render_template, request, send_from_directory
import os
import requests
import json
from werkzeug.utils import secure_filename
from utils import process_image  # Import your custom image processing function

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MLFLOW_API_URL = "http://127.0.0.1:5001/invocations"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mapping numeric predictions to tumor types
class_mapping = {
    'No tumor': 0,
    'Pituitary': 1,
    'Meningioma': 2,
    'Glioma': 3
}

# Reverse mapping for easy lookup
inverse_class_mapping = {v: k for k, v in class_mapping.items()}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image and prepare it for the MLflow API
            try:
                input_data = process_image(filepath)

                # Send the processed data to the MLflow API
                response = requests.post(
                    MLFLOW_API_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(input_data)
                )

                if response.status_code == 200:
                    prediction = response.json()['predictions'][0]  # Extract prediction value
                    tumor_type = inverse_class_mapping.get(prediction, "Unknown")  # Map to tumor type
                    
                    # Use url_for to generate the correct URL for the uploaded image
                    uploaded_image_url = f"/uploads/{filename}"
                    return render_template('index.html', 
                                           uploaded_image=uploaded_image_url, 
                                           prediction=tumor_type)
                else:
                    return render_template('index.html', error="MLflow API Error: " + response.text)
            except Exception as e:
                return render_template('index.html', error=str(e))
        else:
            return render_template('index.html', error="Invalid file type. Only PNG, JPG, and JPEG are allowed.")
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5050)