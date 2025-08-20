import whisperx
import torch


class AudioProcessor:
    def __init__(self):
        """
        Initializes the AudioProcessor and preloads models to RAM (CPU).
        This is a slow, one-time operation that happens on server start.
        """
        print("🚀 Preloading AI models to RAM, this may take a moment...")
        
        # Check for GPU and set devices
        self.device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_cpu = torch.device("cpu")
        
        # --- Preload WhisperX to CPU ---
        # The key is device='cpu'. This forces the model into system RAM.
        self.whisper_model = whisperx.load_model(
            "medium",
            device="cpu",
            compute_type="float32", # Use float32 for CPU for better compatibility
            language="en" # Optional: specify language if known
        )
        
        print("✅ WhisperX model preloaded to RAM.")
        # In the future, you would also preload Pyannote and your LLM here.
        input("Press Enter to exit...")  # Keep the script running for testing

    def transcribe_audio(self, wav_filepath):
        """
        Transcribes an audio file using the preloaded model.
        """
        print("🧠 Moving WhisperX to GPU for transcription...")
        # 1. Move the preloaded model to the GPU (fast)
        self.whisper_model.to(self.device_gpu)
        
        # 2. Transcribe the audio on the GPU
        result = self.whisper_model.transcribe(wav_filepath, batch_size=4)
        
        # 3. Move the model back to the CPU (fast) to free up VRAM
        self.whisper_model.to(self.device_cpu)
        
        print("✅ Transcription complete. Model moved back to CPU.")
        return result['segments'] # Return the transcribed segments
    
audio_pipeline = AudioProcessor()
# Example usage