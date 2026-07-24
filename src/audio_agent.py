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
import hashlib
import json
import uuid
from collections import defaultdict
from datetime import timezone
from audio_preprocessor import AudioPreprocessor
from infrastructure.adapter.ASR_Adapter import WhisperAdapter as asr_adapter
from infrastructure.adapter.pyannoteAdapter import PyannoteAdapter
from infrastructure.adapter.ecapaVoxcelebAdapter import EcapaVoxcelebAdapter
from infrastructure.adapter.SQLiteVoiceAdapter import SQLiteVoiceAdapter
from application.services.system_idle_service import SystemIdleService
from core.models import AmbientEvent, DiarizationResult, InferenceRequest, TranscriptionResult
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
    def __init__(
        self,
        upload_dir: str = str(UPLOAD_DIR),
        voice_db: str = VOICE_DB,
        gpu_lock: threading.Lock = None,
        audio_active_event: threading.Event = None,
        llm_active_event: threading.Event = None,
        capture_store=None,
        capture_control=None,
        autonomy_store=None,
        resource_governor=None,
        asr_model_name: str = "ambient_asr",
        deferred_asr_max_audio_seconds: float = 300.0,
    ):
        self.upload_dir = upload_dir
        self.voice_db = voice_db
        self.transcription_queue = queue.Queue()
        self.gpu_lock = gpu_lock or threading.Lock()
        self.audio_active_event = audio_active_event or threading.Event()
        self.llm_active_event = llm_active_event or threading.Event()
        self.capture_store = capture_store
        self.capture_control = capture_control
        self.autonomy_store = autonomy_store
        self.resource_governor = resource_governor
        self.asr_model_name = str(asr_model_name or "ambient_asr")
        self.deferred_asr_max_audio_seconds = max(30.0, float(deferred_asr_max_audio_seconds))
        self._asr_window_audio_seconds = 0.0
        self._asr_window_started_at = 0.0
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
        logging.info("Audio pipeline is continuously capturing; heavy ASR is resource-gated.")
        while not self.stop_event.is_set():
            if self.capture_control is not None and self.capture_control.is_paused():
                self.audio_active_event.clear()
                if self.stop_event.wait(0.5):
                    break
                continue
            try:
                file_path = self.processing_queue.get(timeout=1)
            except queue.Empty:
                file_path = None

            if file_path is None and self.stop_event.is_set():
                self.processing_queue.task_done()
                break
            if file_path is not None:
                if self.autonomy_store is not None and self.resource_governor is not None:
                    try:
                        self._persist_audio_capture(file_path)
                    except Exception:
                        logging.exception("Failed to persist raw audio capture before ASR: %s", file_path)
                        if Path(file_path).exists() and not self.stop_event.wait(1.0):
                            self.processing_queue.put(file_path)
                    finally:
                        self.processing_queue.task_done()
                else:
                    try:
                        self._process_audio_path(file_path)
                    finally:
                        self.processing_queue.task_done()
                    continue

            if self.autonomy_store is not None and self.resource_governor is not None:
                self._process_deferred_audio_once()
                continue
            if file_path is None:
                continue

        self.audio_active_event.clear()
        self.llm_active_event.clear()
        self.audio_agent.release_all_models()

    def _persist_audio_capture(self, file_path: str) -> None:
        path = Path(file_path)
        self._wait_for_stable_file(path)
        stat = path.stat()
        try:
            info = torchaudio.info(str(path))
            duration = float(info.num_frames / info.sample_rate) if info.sample_rate else 0.0
        except Exception:
            duration = 0.0
        source_ref = str(path)
        if self.capture_store is not None:
            source_ref = self.capture_store.store_file(file_path, kind="audio", delete_source=True)
        if self.autonomy_store is None:
            return
        occurred_at = datetime.now(timezone.utc).isoformat()
        fingerprint = hashlib.sha256(
            f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        self.autonomy_store.enqueue_event(
            AmbientEvent(
                event_id=uuid.uuid4().hex,
                event_type="audio_capture_pending",
                source_kind="audio_capture",
                source_ref=source_ref,
                occurred_at=occurred_at,
                payload_json=json.dumps(
                    {"audio_ref": source_ref, "duration_seconds": duration, "original_name": path.name}
                ),
                confidence=0.7,
                privacy_label="sensitive_audio",
                fingerprint=fingerprint,
                status="pending",
                priority=0.7,
                available_at=occurred_at,
            )
        )

    def _wait_for_stable_file(self, path: Path, *, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        previous_size = -1
        stable_checks = 0
        while time.monotonic() < deadline:
            size = path.stat().st_size
            if size > 0 and size == previous_size:
                stable_checks += 1
                if stable_checks >= 2:
                    return
            else:
                stable_checks = 0
            previous_size = size
            if self.stop_event.wait(0.1):
                raise RuntimeError("audio capture stopped before the segment finished writing")
        raise TimeoutError(f"audio segment did not stabilize before storage: {path}")

    def _process_deferred_audio_once(self) -> None:
        now = time.monotonic()
        if self._asr_window_started_at and (
            now - self._asr_window_started_at >= 90.0
            or self._asr_window_audio_seconds >= self.deferred_asr_max_audio_seconds
        ):
            if now - self._asr_window_started_at < 150.0:
                return
            self._asr_window_started_at = 0.0
            self._asr_window_audio_seconds = 0.0
        event = self.autonomy_store.claim_next_event(
            lease_seconds=600, event_types=["audio_capture_pending"]
        )
        if event is None:
            self._asr_window_started_at = 0.0
            self._asr_window_audio_seconds = 0.0
            return
        request = InferenceRequest(
            workload="deferred_asr",
            model_name=self.asr_model_name,
            background=True,
            user_active=not self.system_idle_service.is_user_idle(),
            priority=70,
        )
        lease = self.resource_governor.request_lease(request)
        if not lease.acquired:
            self.autonomy_store.defer_event(event.event_id, reason=lease.decision.reason, delay_seconds=30)
            return
        payload = json.loads(event.payload_json or "{}")
        audio_ref = str(payload.get("audio_ref") or event.source_ref)
        duration = float(payload.get("duration_seconds") or 0.0)
        if not self._asr_window_started_at:
            self._asr_window_started_at = now
        try:
            self.audio_active_event.set()
            deadline = time.monotonic() + 15.0
            while self.llm_active_event.is_set() and time.monotonic() < deadline:
                if self.stop_event.wait(0.1):
                    self.autonomy_store.defer_event(
                        event.event_id,
                        reason="ASR shutdown interrupted resource acquisition",
                        delay_seconds=15,
                    )
                    return
            if self.llm_active_event.is_set():
                self.autonomy_store.defer_event(
                    event.event_id,
                    reason="resident LLM did not release memory for deferred ASR",
                    delay_seconds=15,
                )
                return
            if audio_ref.startswith("capture://") and self.capture_store is not None:
                with self.capture_store.materialize(audio_ref) as materialized:
                    self._process_audio_path(materialized, archive_after=False)
            else:
                self._process_audio_path(audio_ref, archive_after=False)
            self.autonomy_store.complete_event(event.event_id)
            self._asr_window_audio_seconds += max(0.0, duration)
        except Exception as exc:
            logging.exception("Deferred ASR failed for %s", audio_ref)
            self.autonomy_store.retry_event(event.event_id, error_text=str(exc), delay_seconds=60)
        finally:
            self.audio_agent.release_all_models()
            self.audio_active_event.clear()
            lease.__exit__(None, None, None)

    def _process_audio_path(self, file_path: str, *, archive_after: bool = True) -> None:
        self.audio_active_event.set()
        try:
            with self.gpu_lock:
                self.audio_agent.run(file_path)
        except Exception:
            logging.exception("Failed to process file: %s", file_path)
            raise
        finally:
            if archive_after and self.capture_store is not None and Path(file_path).exists():
                self.capture_store.store_file(file_path, kind="audio", delete_source=True)
            self.audio_agent.release_all_models()
            self.audio_active_event.clear()

    def start_service(self):
        self.stop_event.clear()
        self.observer = Observer()
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        self.observer.schedule(Handler(self.processing_queue), str(self.upload_dir), recursive=False)
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
