from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading
import queue
from pathlib import Path
import os
import torch
import gc
import torchaudio
import logging
from collections import defaultdict
import numpy as np
from audio_preprocessor import AudioPreprocessor
from infrastructure.adapter.ASR_Adapter import WhisperAdapter
from infrastructure.adapter.pyannoteAdapter import PyannoteAdapter
from infrastructure.adapter.ecapaVoxcelebAdapter import EcapaVoxcelebAdapter
from infrastructure.adapter.SQLiteVoiceAdapter import SQLiteVoiceAdapter
from core.models import DiarizationResult, TranscriptionResult

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # helps cuDNN find kernels

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
    def __init__(self, transcription_queue: queue.Queue):
        self.preprocessor = None
        self.asr = None
        self.diarization = None
        self.encoder = None
        self.transcription_queue = transcription_queue

    def preprocess_audio(self, audio_file_path):
        self.preprocessor = AudioPreprocessor()
        return self.preprocessor.run(audio_file_path)
    
    def transcribe_audio(self, audio_file_path: str, vad_filter: bool, word_timestamps: bool, batch_size: int = 8)-> list[TranscriptionResult]:
        self.asr = WhisperAdapter(model_size=HIN2HINGLISH, device="cuda")
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
    
    def compare_embeddings(self, db, diarization_result: list[DiarizationResult], waveform: torch.Tensor | None = None, samplerate: int | None = None):
        self.encoder = EcapaVoxcelebAdapter(db)
        try:
            if waveform is None or samplerate is None:
                waveform, samplerate = torchaudio.load(diarization_result[0].audio_file)
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
        
        transcript_path = os.path.join(TRANSCRIPTIONS_DIR, f"final_transcript_{str(file_counter)}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            for entry in final_transcript:
                f.write(f"[{entry[0]:.4f} - {entry[1]:.4f}] -> {entry[2]}: {entry[3]}\n")

        logging.info(f"Final transcript for-{diarization_result[0].audio_file} saved to final_transcript_{str(file_counter)}.txt")
        self.transcription_queue.put(transcript_path)
        file_counter += 1
    
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
        
        
    def run(self, audio_file: str, skip_preprocessing: bool = False):
        db = self.connect_db(VOICE_DB)
        processed_file = audio_file if skip_preprocessing else self.preprocess_audio(audio_file)
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed audio not found: {processed_file}")
        logging.info(f"Processed file: {processed_file}")
        diarization_result = self.diarize_audio(processed_file)
        transcription_result = self.transcribe_audio(processed_file, vad_filter=False, word_timestamps= True)
        diarization_result = self.compare_embeddings(db, diarization_result)
        self.merge_transciptions_and_diarizations(transcription=transcription_result, diarization_result=diarization_result)
    
    def run_raw_audio(self, audio_bytes: bytes, sample_rate: int = 16000, channels: int = 1, skip_preprocessing: bool = True):
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")
        if channels <= 0:
            raise ValueError("channels must be greater than 0")

        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            raise ValueError("audio_bytes does not contain valid PCM samples")
        if audio_int16.size % channels != 0:
            raise ValueError("audio_bytes sample count is not divisible by channel count")

        waveform = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
        waveform = waveform.reshape(-1, channels).transpose(0, 1).contiguous()

        if channels > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        db = self.connect_db(VOICE_DB)
        if skip_preprocessing:
            self.diarization = PyannoteAdapter(HF_TOKEN)
            try:
                diarization_result = self.diarization.diarize_waveform(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    audio_label="in_memory_audio",
                )
            finally:
                self.unload_model("diarization")
            transcription_result = self.transcribe_audio(
                waveform.squeeze(0).cpu().numpy(),
                vad_filter=False,
                word_timestamps=True,
            )
            diarization_result = self.compare_embeddings(db, diarization_result, waveform=waveform, samplerate=sample_rate)
            self.merge_transciptions_and_diarizations(
                transcription=transcription_result,
                diarization_result=diarization_result,
            )
            return

        raise ValueError("Raw-audio processing currently requires skip_preprocessing=True")
        
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
    def __init__(self, upload_dir: str = UPLOAD_DIR, voice_db: str = VOICE_DB, gpu_lock: threading.Lock = None, audio_active_event: threading.Event = None, llm_active_event: threading.Event= None):
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
        self.worker_thread = threading.Thread(target=self._process_uploads, daemon=True)
    def get_audio_active_event(self) -> threading.Event:
        return self.audio_active_event
    
    def get_transcription_queue(self) -> queue.Queue:
        return self.transcription_queue
    
    def enqueue_audio_file(self, file_path: str):
        self.processing_queue.put(file_path)
    
    def enqueue_raw_audio(self, audio_bytes: bytes, sample_rate: int = 16000, channels: int = 1, skip_preprocessing: bool = True):
        self.processing_queue.put(
            {
                "type": "raw_audio",
                "audio_bytes": audio_bytes,
                "sample_rate": sample_rate,
                "channels": channels,
                "skip_preprocessing": skip_preprocessing,
            }
        )
    
    def _process_uploads(self):
        IDLE_TIMEOUT = 20
        idle_elapsed = 0
        CHECK_INTERVAL = 5
        self.audio_active_event.set()
        self.llm_active_event.clear()
        logging.info("Audio pipeline is active")
        while True:
            if self.llm_active_event.is_set():
                if self.audio_active_event.is_set():
                    self.audio_active_event.clear()
                    logging.info("LLM pipeline active, audio pipeline waiting...")
                idle_elapsed = 0
                while self.llm_active_event.is_set():
                    time.sleep(CHECK_INTERVAL)
                idle_elapsed = 0
                continue

            try:
                item = self.processing_queue.get(timeout=CHECK_INTERVAL)
                idle_elapsed = 0
            except queue.Empty:
                if self.audio_active_event.is_set():
                    idle_elapsed += CHECK_INTERVAL
                    logging.warning(
                        f"No files received. "
                        f"Idle for {idle_elapsed}s / {IDLE_TIMEOUT}s before audio agent shutdown."
                    )
                    if idle_elapsed >= IDLE_TIMEOUT:
                        logging.info(f"Audio pipeline idle for {IDLE_TIMEOUT}s. Shutting down audio agent.")
                        idle_elapsed = 0
                        self.audio_active_event.clear()
                else:
                    idle_elapsed = 0
                continue

            if item is None:
                self.processing_queue.task_done()
                break

            if not self.audio_active_event.is_set():
                self.audio_active_event.set()
                logging.info("Audio Pipeline is active, Ambient Agent will Wait")
            idle_elapsed = 0

            try:
                with self.gpu_lock:
                    if isinstance(item, dict) and item.get("type") == "raw_audio":
                        self.audio_agent.run_raw_audio(
                            audio_bytes=item["audio_bytes"],
                            sample_rate=item.get("sample_rate", 16000),
                            channels=item.get("channels", 1),
                            skip_preprocessing=item.get("skip_preprocessing", True),
                        )
                    else:
                        self.audio_agent.run(item)
            except KeyboardInterrupt:
                logging.info("Terminating")
            except Exception:
                logging.exception(f"Failed to process item: {item}")
            finally:
                self.audio_agent.release_all_models()
                self.processing_queue.task_done()
        self.audio_active_event.clear()
    
    def start_service(self):
        observer = Observer()
        observer.schedule(Handler(self.processing_queue), UPLOAD_DIR, recursive=False)
        self.worker_thread.start()
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        finally:
            observer.join()
            self.processing_queue.put(None)
            self.worker_thread.join()

if __name__ == "__main__":
    agent_service = AudioAgentService()
    agent_service.start_service()
