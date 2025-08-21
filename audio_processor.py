import whisperx
import torch
import warnings

# Suppress specific UserWarnings from torchaudio and pyannote
warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote')
# Suppress the ReproducibilityWarning
warnings.filterwarnings("ignore", message="TensorFloat-32 is disabled", category=UserWarning)



class AudioProcessor:
    def __init__(self):
        """
        Initializes the AudioProcessor and preloads models to VRAM.
        This is a slow, one-time operation that happens on server start.
        """
        print("Loading whisperX in GPU")
        
        # Check for GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA (GPU) is not available. This processor requires a GPU.")

        # --- Preload WhisperX to GPU ---
        # For GPU, float16 is much faster.
        self.whisper_model = whisperx.load_model(
            "medium",
            device="cuda",
            compute_type="float16", 
        )
        
        print("✅ WhisperX model Loaded to VRAM.")

    def transcribe_audio(self, wav_filepath):
        """
        Transcribes an audio file using the preloaded model.
        Specifying the language avoids detection and speeds up transcription.
        """
        # 1. Load audio
        audio = whisperx.load_audio(wav_filepath)

        # 2. Transcribe the audio on the GPU
        # By adding the language parameter, you avoid the "No language specified" warning.
        result = self.whisper_model.transcribe(audio, batch_size=4)
        
        print("✅ Transcription complete.")
        
        # 3. Save transcription to a file
        transcription_path = "transcriptions/transcription.txt"
        with open(transcription_path, "w", encoding="utf-8") as f:
            for segment in result['segments']:
                f.write(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n")
        
        print(f"Transcription saved to {transcription_path}")
        return result['segments'] # Return the transcribed segments

    # Example usage
transciber= AudioProcessor()
transciber.transcribe_audio("uploads/2025-08-22_00-47-04.wav")

