from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path
import os
import torch
import gc
import torchaudio
import logging
from collections import defaultdict
from audio_preprocessor import AudioPreprocessor
from infrastructure.adapter.ASR_Adapter import WhisperAdapter
from infrastructure.adapter.pyannoteAdapter import PyannoteAdapter
from infrastructure.adapter.ecapaVoxcelebAdapter import EcapaVoxcelebAdapter
from infrastructure.adapter.SQLiteVoiceAdapter import SQLiteVoiceAdapter
from core.models import DiarizationResult, TranscriptionResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

current_dir = Path(__file__).parent
project_root = current_dir.parent
os.chdir(project_root)

UPLOAD_DIR = "uploads/"
VOICE_DB = "database/voice_database.db"
TRANSCRIPTIONS_DIR = "transcriptions/"
HIN2HINGLISH = "Hin2Hinglish-ct2/"
CLEANED_AUDIO_DIR = "cleaned_audio/"



HF_TOKEN = os.getenv("HF_TOKEN", None)
MIN_TIME_THRESHOLD = 0.2 #seconds
file_counter = 1
class AudioAgent:
    def __init__(self):
        self.preprocessor = None
        self.asr = None
        self.diarization = None
        self.encoder = None

    def preprocess_audio(self, audio_file_path):
        self.preprocessor = AudioPreprocessor()
        return self.preprocessor.run(audio_file_path)
    
    def transcribe_audio(self, audio_file_path: str, vad_filter: bool, word_timestamps: bool, batch_size: int = 8)-> list[TranscriptionResult]:
        self.asr = WhisperAdapter(model_size=HIN2HINGLISH, device="cuda")
        segments = self.asr.transcribe_audio(audio_file_path, vad_filter, word_timestamps, batch_size)
        self.unload_model(self.asr)
        return segments
    

    def diarize_audio(self,audio_file_path: str) -> list[DiarizationResult]:
        self.diarization = PyannoteAdapter(HF_TOKEN)
        diarization_result = self.diarization.diarize_audio(audio_file_path)
        self.unload_model(self.diarization)
        return diarization_result
    
    def connect_db(self, voice_database_path: str):
        db = SQLiteVoiceAdapter(voice_database_path)
        return db
    
    def compare_embeddings(self, db, diarization_result: list[DiarizationResult]):
        self.encoder = EcapaVoxcelebAdapter(db)
        waveform, samplerate= torchaudio.load(diarization_result[0].audio_file)
        speaker_audio_tensor = defaultdict(list)
        for i in diarization_result:
            segment_duration = i.end_time - i.start_time
            if segment_duration <= MIN_TIME_THRESHOLD:
                continue

            sample_start = int(i.start_time * samplerate)
            sample_end = int(i.end_time * samplerate)
            if sample_end <= sample_start:
                continue

            cropped_tensor = waveform[:, sample_start:sample_end]
            if cropped_tensor.shape[0] > 1:
                cropped_tensor = cropped_tensor.mean(dim=0, keepdim=True)

            speaker_audio_tensor[i.speaker_label].append(cropped_tensor)
            mapping = self.encoder.identify_speaker(
                cropped_tensor,
                original_label=i.speaker_label,
                threshold=0.3,
            )
            i.speaker_label = f"{mapping.identified_label}- [{mapping.score*100:.3f}]"

        self.unload_model(self.encoder)
        return diarization_result
        
    def merge_transciptions_and_diarizations(self, transcription: list[TranscriptionResult], diarization_result: list[DiarizationResult]):
        global file_counter
        if transcription is None:
            logging.error("No transcription object received")
            return
        if diarization_result is None:
            logging.error("No diarization object received")
            return
        TOLERANCE = 0.25
        merged = []
        for i in diarization_result:
            merged.append({
                "start": i.start_time,
                "end": i.end_time,
                "speaker": i.speaker_label,
                "words": []
            })
        
        for segment in transcription:
            for word in segment.word_timestamps:
                word_start = word["start"]
                word_text = word["word"]
                for entry in merged:
                    if entry["start"] - TOLERANCE <= word_start <= entry["end"] + TOLERANCE:
                        entry["words"].append(word_text)
                        break
                        
        final_transcript = []
        for entry in merged:
            if not entry["words"]:
                continue
            sentence = " ".join(entry["words"]).strip()
            final_transcript.append((entry["start"], entry["end"], entry["speaker"], sentence))

        if not final_transcript:
            logging.warning("No words matched diarization turns; transcript not written.")
            return
        
        with open(os.path.join(TRANSCRIPTIONS_DIR, f"final_transcript_{str(file_counter)}.txt"), "w", encoding="utf-8") as f:
            for entry in final_transcript:
                f.write(f"[{entry[0]:.4f} - {entry[1]:.4f}] -> {entry[2]}: {entry[3]}\n")

        logging.info(f"Final transcript for-{diarization_result[0].audio_file} saved to final_transcript_{str(file_counter)}.txt")
        file_counter += 1
    
    def unload_model(self, model_object):
        '''
        Unloads the given model from memory
        Args:
            model_object: Model to be unloaded
        '''
        model_object = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        
    def run(self, audio_file: str):
        db = self.connect_db(VOICE_DB)
        processed_file = self.preprocess_audio(audio_file)
        logging.info(f"Processed file: {processed_file}")
        diarization_result = self.diarize_audio(processed_file)
        transcription_result = self.transcribe_audio(processed_file, vad_filter=True, word_timestamps= True)
        diarization_result = self.compare_embeddings(db, diarization_result)
        self.merge_transciptions_and_diarizations(transcription=transcription_result, diarization_result=diarization_result)
        

agent = AudioAgent()
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print("New file:", event.src_path)
            base_name= os.path.splitext(os.path.basename(event.src_path))[0]
            file_path = os.path.join(CLEANED_AUDIO_DIR, f"{base_name}_final.wav")
            agent.run(file_path)

class AudioAgentService:
    def __init__(self):
        pass
    
    def start_service(self):
        observer = Observer()
        observer.schedule(Handler(), UPLOAD_DIR, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()

if __name__ == "__main__":
    agent_service = AudioAgentService()
    agent_service.start_service()