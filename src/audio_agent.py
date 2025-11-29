from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path
import os
from audio_preprocessor import AudioPreprocessor
from ASR_Model import ASR, HIN2HINGLISH

current_dir = Path(__file__).parent
project_root = current_dir.parent
os.chdir(project_root)

UPLOAD_DIR = "uploads/"
cleaned_audio = "cleaned_audio/"
preprocessor = AudioPreprocessor()
asr = ASR(model_size=HIN2HINGLISH, device="cuda", compute_type="int8")

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print("New file:", event.src_path)
            base_name= os.path.splitext(os.path.basename(event.src_path))[0]
            file_path = os.path.join(cleaned_audio, f"{base_name}_final.wav")
            preprocessor.run(input_file=event.src_path)
            preprocessor.unload_model()
            os.remove(event.src_path)
            asr.run(file_path)
            asr.unload_model()
            
            
observer = Observer()
observer.schedule(Handler(), UPLOAD_DIR, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()