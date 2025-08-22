import whisperx
import torch
import warnings
import os
from whisperx.diarize import DiarizationPipeline

warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote')
# Suppress the ReproducibilityWarning
warnings.filterwarnings("ignore", message="TensorFloat-32 is disabled", category=UserWarning)

HF_TOKEN = "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"
class AudioProcessor:
    def __init__(self):
        """
        Initializes the AudioProcessor. The model is loaded on-demand with the first transcription request.
        """
        print("AudioProcessor initialized. Model will be loaded on first use.")
        self.whisper_model = None
        # Check for GPU availability early.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA (GPU) is not available. This processor requires a GPU.")

    def _load_model(self):
        """
        Loads the WhisperX model into VRAM. This is a slow, one-time operation.
        """
        print("Loading WhisperX model to VRAM (this may take a moment)...")
        self.whisper_model = whisperx.load_model(
            "medium",
            device="cuda",
            compute_type="float16", 
        )
        self.model_a, self.metadata=whisperx.load_align_model(language_code="hi", device="cuda")
        self.diarize_model=DiarizationPipeline(use_auth_token=HF_TOKEN, device="cuda")
        
        print("✅ Models loaded successfully.")

    def transcribe_audio(self, wav_filepath, language=""):
        """
        Transcribes an audio file using the preloaded model.
        If the model is not loaded, it will be loaded before transcription.
        """
        # 1. Load the model if it hasn't been loaded yet.
        if self.whisper_model is None:
            self._load_model()

        # 2. Load audio
        print(f"Starting transcription for: {wav_filepath}")
        audio = whisperx.load_audio(wav_filepath)
        
        # 3. Transcribe the audio on the GPU
        result = self.whisper_model.transcribe(audio, batch_size=8, language=language)
        print("✅ Transcription complete.")
        
        result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, device="cuda")
        
        diarize_segments = self.diarize_model(audio)
        print(diarize_segments)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write(str(result));
            
        
        # 3. Save transcription to a file
        transcribed_file_name=wav_filepath.rsplit('\\', 1)[-1].rsplit('.', 1)[0]
        transcription_path = f"{transcribed_file_name}_transcription.txt"
        with open(os.path.join("transcriptions",transcription_path), "w", encoding="utf-8") as f:
            for segment in result['segments']:
                f.write(f"[{segment['start']:.3f}s -> {segment['end']:.3f}s] [{segment['speaker']}] {segment['text']}\n")
        
        print(f"Transcription saved to {transcription_path}")
        # return result['segments'] # Return the transcribed segments

    # Example usage
transciber= AudioProcessor()
transciber.transcribe_audio("uploads\\testing.opus")
