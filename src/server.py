from flask import Flask, request, template_rendered
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from audio_preprocessor import convert_audio_to_wav, add_audio_to_queue
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'aac', 'm4a', 'opus'}

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/home")
def home():
    """Serves the home page with a link to the upload form."""
    return '''
        <p>Welcome to the Audio Transcription Service!</p>
        <p>Upload an audio file to begin: <a href="/upload-form">Click here</a></p>
    '''

@app.route("/uploads", methods=["POST"]) 
def upload_file():
    """Handles the file upload, saves the file, and starts the transcription process in a background thread."""
    if 'file' not in request.files:
        return "<p>No file part in the request.</p>", 400
    
    file = request.files['file']
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    extension= file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    filename = f"{timestamp}.{extension}"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Convert the uploaded audio file to WAV format, run this in a separate thread
        threading.Thread(target=add_audio_to_queue, args=(os.path.join(app.config['UPLOAD_FOLDER'], filename),)).start()
        
        
        return f'''<p>File uploaded successfully!</p>
        <p>File name: {filename}</p>
        <a href="/upload-form">Upload another file</a>
        <p>Go to <a href="/">Home</a></p>
        '''
    else:
        return f'''
            <p>Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}</p>
            <p><a href="/upload-form">Try again</a></p>
        ''', 400

@app.route("/upload-form")
def show_upload_form():
    """Displays the HTML form for file uploads."""
    return '''
       <!doctype html>
       <title>Upload an Audio File</title>
       <h1>Upload a New Audio File </h1>
       <form method="post" action="/uploads" enctype="multipart/form-data">
         <input type="file" name="file">
         <input type="submit" value="Upload">
       </form>
    '''
    
if __name__ == '__main__':
    # Ensure the 'uploads' and 'transcriptions' directories exist.
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('transcriptions', exist_ok=True)
    
    # The host='0.0.0.0' makes the server accessible on your local network.
    app.run(host='0.0.0.0', port=5000, debug=True)