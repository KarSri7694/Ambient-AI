from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from datetime import datetime
import threading
import queue
from pathlib import Path
import os
import torch
import gc
import torchaudio
import logging
from collections import defaultdict
from audio_preprocessor import AudioPreprocessor
from infrastructure.adapter.ASR_Adapter import WhisperAdapter as asr_adapter
from infrastructure.adapter.pyannoteAdapter import PyannoteAdapter
from infrastructure.adapter.ecapaVoxcelebAdapter import EcapaVoxcelebAdapter
from infrastructure.adapter.SQLiteVoiceAdapter import SQLiteVoiceAdapter
from application.services.system_idle_service import SystemIdleService
from core.models import DiarizationResult, TranscriptionResult
from config import CONFIG

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # helps cuDNN find kernels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

current_dir = Path(__file__).parent
project_root = current_dir.parent

USER_DATA_DIR = Path(CONFIG.get_str("runtime", "user_data_dir", "D:\\USER_DATA"))
UPLOAD_DIR = Path(CONFIG.get_str("audio", "uploads_dir", str(USER_DATA_DIR / "uploads")))
VOICE_DB = CONFIG.get_str("audio", "voice_db", "database/voice_database.db")
TRANSCRIPTIONS_DIR = Path(CONFIG.get_str("audio", "transcriptions_dir", str(USER_DATA_DIR / "transcriptions")))
HIN2HINGLISH = CONFIG.get_str("audio", "hin2hinglish_model", "Hin2Hinglish-ct2/")
CLEANED_AUDIO_DIR = Path(CONFIG.get_str("audio", "cleaned_audio_dir", str(USER_DATA_DIR / "cleaned_audio")))
TEMP_AUDIO_DIR = Path(CONFIG.get_str("audio", "temp_audio_dir", str(USER_DATA_DIR / "temp_audio")))
USER_IDLE_THRESHOLD_SECONDS = CONFIG.get_int("audio", "user_idle_threshold_seconds", 20)
ALWAYS_ON_MODE = CONFIG.get_bool("runtime", "always_on", False)

HF_TOKEN = CONFIG.get_str("audio", "hf_token", "").strip() or None
MIN_TIME_THRESHOLD = CONFIG.get_float("audio", "min_time_threshold", 0.2)

if not UPLOAD_DIR.exists():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not TRANSCRIPTIONS_DIR.exists():
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

if not CLEANED_AUDIO_DIR.exists():
    CLEANED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

if not TEMP_AUDIO_DIR.exists():
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

class AudioAgent:
    def __init__(self, transcription_queue: queue.Queue):
        self.preprocessor = None
        self.asr = None
        self.diarization = None
        self.encoder = None
        self.transcription_queue = transcription_queue

    def preprocess_audio(self, audio_file_path):
        self.preprocessor = AudioPreprocessor(temp_audio_dir=str(TEMP_AUDIO_DIR), cleaned_audio_dir=str(CLEANED_AUDIO_DIR))
        return self.preprocessor.run(audio_file_path)
    
    def transcribe_audio(self, audio_file_path: str, vad_filter: bool, word_timestamps: bool, batch_size: int = 8)-> list[TranscriptionResult]:
        self.asr = asr_adapter(model_size=HIN2HINGLISH, device="cuda")
        try:
            return self.asr.transcribe_audio(audio_file_path, vad_filter, word_timestamps, batch_size)
        finally:
            self.unload_model("asr")
    

    def diarize_audio(self,audio_file_path: str) -> list[DiarizationResult]:
        self.diarization = PyannoteAdapter(HF_TOKEN)
        try:
            return self.diarization.diarize_audio(audio_file_path)
        finally:
            self.unload_model("diarization")
    
    def connect_db(self, voice_database_path: str):
        db = SQLiteVoiceAdapter(voice_database_path)
        return db
    
    def compare_embeddings(self, db, diarization_result: list[DiarizationResult]):
        self.encoder = EcapaVoxcelebAdapter(db)
        try:
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
                i.speaker_label = f"{mapping.identified_label}- [{mapping.score*100:.3f}%]"

            return diarization_result
        finally:
            self.unload_model("encoder")

    def release_all_models(self):
        """Best-effort cleanup for all model handles after each file."""
        self.preprocessor = None
        self.asr = None
        self.diarization = None
        self.encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def merge_transciptions_and_diarizations(self, transcription: list[TranscriptionResult], diarization_result: list[DiarizationResult]):
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
        
        transcript_name = f"transcript_{datetime.now().strftime('%d%m%Y_%H%M%S')}.txt"
        transcript_path = TRANSCRIPTIONS_DIR / transcript_name
        with open(transcript_path, "w", encoding="utf-8") as f:
            for entry in final_transcript:
                f.write(f"[{entry[0]:.4f} - {entry[1]:.4f}] -> {entry[2]}: {entry[3]}\n")

        logging.info(f"Final transcript for-{diarization_result[0].audio_file} saved to {transcript_name}")
        self.transcription_queue.put(str(transcript_path))
    
    def unload_model(self, attr_name: str):
        '''
        Unloads the given model from memory
        Args:
            model_object: Model to be unloaded
        '''
        setattr(self, attr_name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        
    def run(self, audio_file: str):
        db = self.connect_db(VOICE_DB)
        processed_file = self.preprocess_audio(audio_file)
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed audio not found: {processed_file}")
        logging.info(f"Processed file: {processed_file}")
        diarization_result = self.diarize_audio(processed_file)
        transcription_result = self.transcribe_audio(processed_file, vad_filter=False, word_timestamps= True)
        diarization_result = self.compare_embeddings(db, diarization_result)
        self.merge_transciptions_and_diarizations(transcription=transcription_result, diarization_result=diarization_result)
        
class Handler(FileSystemEventHandler):
    def __init__(self, processing_queue: queue.Queue):
        super().__init__()
        self.processing_queue = processing_queue

    def on_created(self, event):
        if event.is_directory:
            return

        logging.info(f"New file: {event.src_path}")
        self.processing_queue.put(event.src_path)

class AudioAgentService:
    def __init__(self, upload_dir: str = str(UPLOAD_DIR), voice_db: str = VOICE_DB, gpu_lock: threading.Lock = None, audio_active_event: threading.Event = None, llm_active_event: threading.Event= None):
        self.upload_dir = upload_dir
        self.voice_db = voice_db
        self.transcription_queue = queue.Queue()
        self.gpu_lock = gpu_lock or threading.Lock()
        self.audio_active_event = audio_active_event or threading.Event()
        self.llm_active_event = llm_active_event or threading.Event()
        self.audio_agent = AudioAgent(
            transcription_queue=self.transcription_queue
            )
        self.processing_queue = queue.Queue()
        self.system_idle_service = SystemIdleService(
            idle_threshold_seconds=USER_IDLE_THRESHOLD_SECONDS,
        )
        self.stop_event = threading.Event()
        self.observer: Observer | None = None
        self.worker_thread = threading.Thread(target=self._process_uploads, name="AudioUploadWorker")

    def get_audio_active_event(self) -> threading.Event:
        return self.audio_active_event
    
    def get_transcription_queue(self) -> queue.Queue:
        return self.transcription_queue
    
    def _drain_queue(self, target_queue: queue.Queue) -> None:
        while True:
            try:
                item = target_queue.get_nowait()
            except queue.Empty:
                break
            else:
                target_queue.task_done()

    def _process_uploads(self):
        logging.info("Audio pipeline is waiting for uploads and user idle.")
        while not self.stop_event.is_set():
            if not ALWAYS_ON_MODE and not self.system_idle_service.is_user_idle():
                self.audio_active_event.clear()
                if self.stop_event.wait(1):
                    break
                continue

            try:
                file_path = self.processing_queue.get(timeout=1)
            except queue.Empty:
                continue

            if file_path is None:
                self.processing_queue.task_done()
                break

            self.audio_active_event.set()
            self.llm_active_event.clear()
            if ALWAYS_ON_MODE:
                logging.info("Audio file received. ASR pipeline is active in always_on mode.")
            else:
                logging.info("Audio file received while system is idle. ASR pipeline is active.")

            try:
                with self.gpu_lock:
                    self.audio_agent.run(file_path)
            except Exception:
                logging.exception(f"Failed to process file: {file_path}")
            finally:
                self.audio_agent.release_all_models()
                self.audio_active_event.clear()
                self.processing_queue.task_done()
                logging.info("ASR pipeline finished. Returning to idle wait.")
        self.audio_active_event.clear()
        self.llm_active_event.clear()
        self.audio_agent.release_all_models()
    
    def start_service(self):
        self.stop_event.clear()
        self.observer = Observer()
        self.observer.schedule(Handler(self.processing_queue), str(UPLOAD_DIR), recursive=False)
        self.worker_thread.start()
        self.observer.start()
        
        try:
            while not self.stop_event.wait(1):
                pass
        finally:
            self.stop_service()

    def stop_service(self, join_timeout: float = 10.0) -> None:
        self.stop_event.set()
        if self.observer is not None:
            try:
                self.observer.stop()
            finally:
                self.observer.join(timeout=join_timeout)
                self.observer = None
        self.processing_queue.put(None)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=join_timeout)
        self.audio_agent.release_all_models()
        self.audio_active_event.clear()
        self.llm_active_event.clear()
        self._drain_queue(self.processing_queue)
        self._drain_queue(self.transcription_queue)

if __name__ == "__main__":
    agent_service = AudioAgentService()
    agent_service.start_service()
