import ffmpeg;
import os
from audio_processor import AudioProcessor

base=""
extension=".wav"

# Function to convert audio files to WAV format
def convert_audio_to_wav(input_file):
    base= input_file.rsplit('.', 1)[0]
    try:
        ffmpeg.input(input_file).output(base+extension, ar=16000, ac=1, format='wav').run(overwrite_output=True)
        print(f"Converted {input_file} to {input_file}.wav")
        print(f"Sending {base+extension} to Whisper for transcription...")
        #send voice to whisper for transcription
        h=AudioProcessor().transcribe_audio(base+extension)
    except ffmpeg.Error as e:
        print(f"Error converting file: {e}")
        return False
    return True 

def whisper_transcribe(input_file):
    AudioProcessor().transcribe_audio(base+extension)
    
# Example usage
