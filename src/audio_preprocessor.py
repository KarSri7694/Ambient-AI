import ffmpeg;
import os
import pyloudnorm as pyln
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch


'''
Process to convert raw audio to cleaned audio:
1. Convert to WAV format with 16kHz sample rate 
2. Reduce noise using Demucs, convert to mono audio after noise reduction
3. Normalize audio to -23 LUFS
'''

class AudioPreprocessor:
    """
    A class to preprocess audio files by converting them to WAV, reducing noise, and normalizing loudness.
    """
    def __init__(self, input_queue_file="audio_queue.txt", temp_audio_dir="temp_audio",isSample=False):
        """
        Initializes the AudioPreprocessor with specified directories and model.

        Args:
            input_queue_file (str, optional): Path to the text file containing a list of audio files to process. Defaults to "audio_queue.txt".
            temp_audio_dir (str, optional): Path to the directory for temporary audio files. Defaults to "temp_audio".
            isSample (bool, optional): Flag to indicate if the audio is a voice sample. Defaults to False.
        """
        self.extension=".wav"
        self.device_cpu = 'cpu'
        self.device_gpu = 'cuda'
        self.TEMP_AUDIO_DIR = temp_audio_dir
        self.input_queue_file = input_queue_file
        self.isSample = isSample
        self.VOICE_SAMPLES_DIR = "voice_samples"
        self.OUTPUT_DIR = "output"
        self.CONVERTED_SUFFIX = "_converted"
        self.NOISE_REDUCED_SUFFIX = "_cleaned"
        self.NORMALIZED_SUFFIX = "_final"
        
        #load Demucs model for noise reduction
        self.model = get_model('htdemucs_ft')
        self.model.to(self.device_gpu if torch.cuda.is_available() else self.device_cpu)
        
        #create necessary directories if they don't exist
        os.makedirs(self.TEMP_AUDIO_DIR, exist_ok=True)
        os.makedirs(self.VOICE_SAMPLES_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def run(self): 
        """
        Runs the audio preprocessing pipeline for each file in the input queue.
        """
        #loop through the audio queue and process each file
        lines = []
        with open(self.input_queue_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            input_file = line.strip()
            line=""
            
            base_name= os.path.splitext(os.path.basename(input_file))[0]
            converted_file = os.path.join(self.TEMP_AUDIO_DIR, f"{base_name}_converted.wav")
            noise_reduced_file = os.path.join(self.TEMP_AUDIO_DIR, f"{base_name}_cleaned.wav")
            final_output_dir = self.VOICE_SAMPLES_DIR if self.isSample else self.OUTPUT_DIR
            final_file = os.path.join(final_output_dir, f"{base_name}_final.wav")
            
            
            if os.path.exists(input_file):
                success = self.convert_audio_to_wav(input_file, converted_file)
                if success:
                    print(f"Converted: {input_file} to WAV")
                    noise_reduced = self.reduce_noise(converted_file, noise_reduced_file)
                    if noise_reduced:
                        print(f"Noise reduced for: {input_file}")
                        normalized = self.normalize_audio(noise_reduced_file, final_file)
                        if normalized is not False:
                            print(f"Normalized audio saved for: {input_file}")
                        else:
                            print(f"Failed to normalize audio for: {input_file}")
                    else:
                        print(f"Failed to reduce noise for: {input_file}")
                else:
                    print(f"Failed to convert: {input_file}")
            else:
                print(f"File does not exist: {input_file}")    
        
    def convert_audio_to_wav(self,input_file, output_file):
        '''
        Convert audio files to WAV format with 16kHz sample rate and 2 channels
        Args:
        input_file (str): Path to the input audio file
        output_file (str): Path to the output WAV file
        Returns:
        bool: True if conversion is successful, False otherwise'''
        
        try:
            ffmpeg.input(input_file).output(os.path.join(output_file), ar=16000, ac=2, format='wav').run(overwrite_output=True)
            print(f"Converted {input_file} to {input_file}.wav")
            
        except ffmpeg.Error as e:
            print(f"Error converting file: {e}")
            return False
        return True

    def reduce_noise(self,input_file, output_file):
        '''
        Reduce noise from audio using Demucs model
        Args:
        input_file (str): Path to the input WAV file
        Returns:
        bool: True if noise reduction is successful, False otherwise
        '''
        try:
            # Load audio file
            wav, sr = sf.read(input_file)
            wav_tensor = torch.from_numpy(wav.T).float()
            wav_tensor = wav_tensor.unsqueeze(0) # add batch dimension
            wav_tensor = wav_tensor.to(self.device_gpu if torch.cuda.is_available() else self.device_cpu)
            separated_sources= apply_model(self.model, wav_tensor, device=self.device_gpu, shifts=5, progress=True)[0] # assuming single channel input
            cleaned_audio = separated_sources[3]
            
            cleaned_audio_mono = torch.mean(cleaned_audio, dim=0) # convert to mono by averaging channels
            
            #save to temp folder
            sf.write(os.path.join(output_file), cleaned_audio_mono.cpu().numpy().T, sr)
            return True
        except Exception as e:
            print(f"Error in noise reduction: {e}")
            return False        
        
    def normalize_audio(self,input_file, output_file, target_lufs=-23.0):
        '''
        Normalize audio loudness to a target LUFS.

        Args:
            input_file (str): Path to the input audio file.
            output_file (str): Path to the output normalized audio file.
            target_lufs (float, optional): Target loudness in LUFS. Defaults to -23.0.
        
        Returns:
            bool: True if normalization is successful, False otherwise.
        '''
        # Load audio file
        try:
            data, rate = sf.read(input_file)
            if data.ndim ==1:
                data = data.reshape(-1, 1)
            # Measure loudness
            meter = pyln.Meter(rate)  # create BS.1770 meter
            loudness = meter.integrated_loudness(data)
            # Normalize audio to target dBFS
            normalized_audio = pyln.normalize.loudness(data, loudness, target_lufs)
            # Save normalized audio back to file
            sf.write(output_file, normalized_audio, rate)
        except Exception as e:
            print(f"Error in normalization: {e}")
            return False

