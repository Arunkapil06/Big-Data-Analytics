from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DATASET_EXTENSIONS = {'csv', 'json', 'xlsx'}

# Create upload folders if they don't exist
os.makedirs(os.path.join(UPLOAD_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'datasets'), exist_ok=True)

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_dataset_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DATASET_EXTENSIONS 