from flask import Flask, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'aac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/home")
def hello_world():
    return "<p>Hello, World!</p>\n "

@app.route("/uploads", methods=["POST"]) 
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return "<p>File uploaded successfully!</p>\n"
    return "<p>File upload failed.</p>\n"

@app.route("/upload-form")
def show_upload_form():
    # This HTML form will send a POST request to your /uploads route
    return '''
       <!doctype html>
       <title>Upload a File</title>
       <h1>Upload a New File</h1>
       <form method="post" action="/uploads" enctype="multipart/form-data">
         <input type="file" name="file">
         <input type="submit" value="Upload">
       </form>
    '''