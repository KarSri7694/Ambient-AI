from flask import Flask, request, template_rendered
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from audio_preprocessor import convert_audio_to_wav

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'aac', 'm4a', 'opus'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/home")
def hello_world():
    return '''<p>Hello, World!</p>\n 
    <p>Upload a file: <a href="/upload-form">Click here</a></p>
    '''

@app.route("/uploads", methods=["POST"]) 
def upload_file():
    file = request.files['file']
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extension= file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    filename = f"{timestamp}.{extension}"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Convert the uploaded audio file to WAV format
        convert_audio_to_wav(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        return f'''<p>File uploaded successfully!</p>
        <p>File name: {filename}</p>
        <a href="/upload-form">Upload another file</a>
        <p>Go to <a href="/">Home</a></p>
        '''
    else:
        return '''<p>Invalid file type. Please upload a valid audio file.</p>
        '''

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
    
if __name__ == '__main__':
    # The host='0.0.0.0' makes the server accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)