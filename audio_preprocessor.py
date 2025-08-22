import ffmpeg;
import os
from whisper_transcriber import AudioProcessor

base=""
extension=".wav"

def add_audio_to_queue(input_file):
    with open("audio_queue.txt", "a") as f:
        f.write(f"{input_file}\n")
    start_conversion()
    
def start_conversion():
    lines = []
    with open("audio_queue.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        input_file = line.strip()
        if os.path.exists(input_file):
            success = convert_audio_to_wav(input_file)
            if success:
                print(f"Processed and converted: {input_file}")
            else:
                print(f"Failed to process: {input_file}")
        else:
            print(f"File does not exist: {input_file}")

# Function to convert audio files to WAV format
def convert_audio_to_wav(input_file):
    base= input_file.rsplit('.', 1)[0]
    try:
        ffmpeg.input(input_file).output(base+extension, ar=16000, ac=1, format='wav').run(overwrite_output=True)
        print(f"Converted {input_file} to {input_file}.wav")
        print(f"Sending {base+extension} to Whisper for transcription...")
        #send voice to whisper for transcription
        whisper_transcribe(base+extension)
    except ffmpeg.Error as e:
        print(f"Error converting file: {e}")
        return False
    return True 

def whisper_transcribe(input_file):
    AudioProcessor().transcribe_audio(input_file)
    
# Example usage
