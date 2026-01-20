from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from speechbrain.inference.speaker import EncoderClassifier
import sqlite3
from silero_vad import load_silero_vad, get_speech_timestamps
import hashlib
from collections import defaultdict
import warnings
import torch
import io
from pathlib import Path
import os
import logging
# import torchaudio without warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')
    warnings.filterwarnings("ignore", category=FutureWarning, module='torchaudio')    
    import torchaudio
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
#Whisper model sizes
SMALL = "small"
MEDIUM = "medium"
LARGE = "large-v3"
TURBO = "turbo"
HIN2HINGLISH = "Hin2Hinglish-ct2/"
HIN2HINGLISH_LARGE = "Hin2Hinglish-Large-ct2/"

#Model identifiers
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
VAD_MODEL = "silero_vad"
EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

#File paths
VOICE_DATABASE = "database\\voice_database.db"
HF_TOKEN_FILE = "HFToken.txt"
TRANSCRIPTIONS_DIR = "transcriptions"

#Thresholds
MIN_TIME_THRESHOLD = 0.2 #seconds
SIMILARITY_THRESHOLD = 0.1 #cosine similarity threshold
UPDATE_THRESHOLD = 0.9 #cosine similarity threshold
CREATE_NEW_EMBEDDING_THRESHOLD = 0.1 #cosine similarity threshold

#Audio chunking parameters
CHUNK_SECONDS = 20 * 60    # 20 min chunks
CHUNK_OVERLAP = 2          # 2 sec overlap

#File counter
file_counter = 1

#set working directory to project root
current_dir = Path(__file__).parent
project_root = current_dir.parent
os.chdir(project_root)


class ASR:
    '''
    Automatic Speech Recognition (ASR) model that transcribes and diarises audio files, and identifies speakers using voice embeddings.
    '''
    def __init__(self, model_size = TURBO, device = "cuda", compute_type = "int8_float16"):
        '''
        Initialize the ASR model with specified parameters.
        Args:
            model_size (str): Size of the Whisper model to use. Options are "small", "medium", "large-v3", "turbo". Default is "turbo".
            device (str): Device to run the model on. Options are "cuda" or "cpu". Default is "cuda".
            compute_type (str): Type of computation to use. Options are "int8_float16
        '''
        self.HFToken = self.load_HFToken()
        self.model_size = model_size
        self.transcribe_model = None
        self.diarisation_model = None
        self.vad_model = None
        self.encoder = None
        self.diarizied_segments = None

    @staticmethod
    def load_HFToken():
        '''
        Load the Hugging Face Token form HFToken.txt
        '''
        with open("HFToken.txt" ,"r") as f:
            return f.read().strip("\n")

    def transcribe_audio(self, vad_filter = False, task = "transcribe", audio_file=None, word_timestamps=False, compute_type="int8_float16",device="cuda",batch_size=8):
        '''
        Transcribe the audio file using the Whisper model.
        Args:
            vad_filter (bool): Whether to apply Voice Activity Detection (VAD) filtering. Default is False.
            task (str): Task to perform, either "transcribe" or "translate". Default is "transcribe".
            language (str): Language code for the audio. Default is "hi" (Hindi).
            audio_file (str): Path to the audio file to transcribe.
        Returns:
            segments (generator object): List of transcription segments with text, start time, and end time.
        '''
        if self.transcribe_model is None:
            self.transcribe_model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        batched_model = BatchedInferencePipeline(self.transcribe_model)
        segments, _ = batched_model.transcribe(audio_file, vad_filter=vad_filter, task=task,word_timestamps=word_timestamps, chunk_length=30, batch_size=batch_size)   
        self.transcribe_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return segments
    
    def diarise_audio(self, audio_file=None):
        '''
        Diarise the audio file using VAD and pyannote.audio's diarisation model.
        Args:
            audio_file (str): Path to the audio file to diarise.
        '''
        if self.diarisation_model is None:
            self.diarisation_model= Pipeline.from_pretrained(DIARIZATION_MODEL, token=self.HFToken)
        if self.vad_model is None:
            self.vad_model = load_silero_vad()
        silero_segmentation = Annotation()
        waveform, sample_rate = torchaudio.load(audio_file)
        
        audio_data = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        speech_timestamps = get_speech_timestamps(waveform, model=self.vad_model, return_seconds=True)
        
        for ts in speech_timestamps:
            start_sec = ts['start']
            end_sec = ts['end']
            segment = Segment(start_sec, end_sec)
            silero_segmentation[segment] = "speech"
        
        self.diarisation_model.to(torch.device("cuda"))
        self.diarizied_segments = self.diarisation_model(audio_data)
        logging.info("Diarization Results:")
        logging.info(self.diarizied_segments.speaker_diarization)
        logging.info(type(self.diarizied_segments))
        logging.info(type(self.diarizied_segments.speaker_diarization))
        return self.diarizied_segments
        
    def connect_db(self):
        '''
        Connect to the SQLite database and create the VOICE_EMBEDDINGS table if it doesn't exist.
        '''
        try:
            conn = sqlite3.connect(VOICE_DATABASE)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS VOICE_EMBEDDINGS (
                            uid INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            embedding BLOB NOT NULL,
                            embedding_hash TEXT NOT NULL UNIQUE,
                            audio BLOB
                    )''')
            conn.commit()
            return conn
        except sqlite3.OperationalError as e:
            logging.error(f"Error connecting to database: {e}")
        
    def compare_embeddings(self, audio_file=None):
        '''
        Compare the embeddings of the diarised segments with the embeddings stored in the database.
        Returns a mapping of speaker labels to identified names.
        Args:
            audio_file (str): Path to the audio file to process.
        Returns:
            speaker_mapping (dict): Mapping of speaker labels to identified names.
        '''
        if self.encoder is None:
            self.encoder = EncoderClassifier.from_hparams(source="D:\\Projects\\ambient_ai\\spkrec-ecapa-voxceleb", run_opts={"device": "cuda"})
        waveform, samplerate = torchaudio.load(audio_file)
        total_sample = waveform.shape[1]
        speaker_audio_tensor = defaultdict(list)
        speaker_probabilities_list = []
        test=[]
        #TO REFACTOR THE CODE
        #currently the code combines audio segments for each speaker into a single tensor which is not the best approach
        #need to process each segment individually and compare embeddings one by one and skip segments that are too short
        #speaker_tensors contains combined tensors for each speaker it should be of some form of list of tensors instead along with time info 
        #over the top of my mind i think of a list of tuples (start_time, end_time, speaker_label, tensor) for each speaker and process them one by one
        speaker_mapping = []
        for turn, _, speaker in self.diarizied_segments.speaker_diarization.itertracks(yield_label=True):
            segment_duration = turn.end - turn.start
            if segment_duration > MIN_TIME_THRESHOLD:
                start_sample = int(turn.start*samplerate)
                end_sample = int(turn.end*samplerate)
                cropped_tensor = waveform[:, start_sample:end_sample]
                test.append((turn.start, turn.end, speaker, cropped_tensor))
                speaker_audio_tensor[speaker].append(cropped_tensor)
        
        speaker_tensors = {
            speaker : torch.cat(tensor_list, dim=1)
            for speaker, tensor_list in speaker_audio_tensor.items()
        }
        # print(test)
        #using SpeechBrain's embedding model to crate an embedding from audio tensor
        with self.connect_db() as conn:
            cursor = conn.cursor()
            for _,_,speaker, tensor in test:
                highest_similiarity_score = 0
                best_name = "UNKNOWN"
                tensor = tensor.to(torch.device("cuda"))
                embedding= self.encoder.encode_batch(tensor)
                embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=2) #normalized embedding
                embedding_norm = embedding_norm.to(torch.device("cuda"))
                # embedding shape = (1, 1, 192)
                
                for row in cursor.execute("SELECT name, embedding FROM VOICE_EMBEDDINGS"):
                    name, embedding_blob =row[0], row[1] 
                    buffer = io.BytesIO(embedding_blob)
                    stored_embedding = torch.load(buffer)
                    stored_embedding_norm = torch.nn.functional.normalize(stored_embedding, p=2, dim=2)
                    stored_embedding_norm = stored_embedding_norm.to(torch.device("cuda"))
                    similiarity_score = torch.nn.functional.cosine_similarity(embedding, stored_embedding, dim=2).item()
                    if similiarity_score > highest_similiarity_score:
                        highest_similiarity_score = similiarity_score
                        best_name = name
                        speaker_probabilities_list.append((speaker, best_name, highest_similiarity_score))
                        
                if highest_similiarity_score > SIMILARITY_THRESHOLD:
                    speaker_mapping.append((speaker, f"{best_name} ({(highest_similiarity_score)*100:.2f}%)"))
                else:
                    speaker_mapping.append((speaker, f"UNKNOWN_{best_name} ({(highest_similiarity_score)*100:.2f}%)"))
           
        # If no speakers were identified, return an empty mapping
        if not speaker_mapping:
            print("\nCould not identify any speakers.")
            return
        print(speaker_probabilities_list)
        print(speaker_mapping)
        return speaker_mapping
                    
    def create_embeddings(self, audio_file=None):
        '''
        Create embeddings for the entire audio file and store it in the database.
        Args:
            audio_file (str): Path to the audio file to process.
        '''
        if self.encoder is None:
            EncoderClassifier.from_hparams(source="D:\\Projects\\ambient_ai\\spkrec-ecapa-voxceleb", run_opts={"device": "cuda"} )
        buffer = io.BytesIO()
        waveform, samplerate = torchaudio.load(audio_file)
        with self.connect_db() as conn:
            cursor = conn.cursor()
            embedding= self.encoder.encode_batch(waveform)
            torch.save(embedding, buffer)
            embedding_blob = buffer.getvalue()
            embedding_hash = hashlib.sha256(embedding_blob).hexdigest()
            cursor.execute("INSERT INTO VOICE_EMBEDDINGS (name, embedding, embedding_hash) VALUES (?, ?, ?)", ("", embedding_blob, embedding_hash))
            conn.commit()
                              
    def merge_transcriptions_and_diarization(self, trancription_segments, speaker_mapping=None, audio_file=None):
        '''
        Merge the transcriptions with the diarisation results to produce a final transcript with speaker labels.
        Args:
            trancription_segments (generator object): List of transcription segments with text, start time, and end time.
            speaker_mapping (list): Mapping of speaker labels to identified names.
            audio_file (str): Path to the audio file.
        '''
        global file_counter

        if self.diarizied_segments is None:
            print("Diarisation has not been performed yet.")
            return

        # Build a quick lookup for speaker labels -> names.
        speaker_lookup = {speaker: name for speaker, name in (speaker_mapping or [])}

        # Freeze generators so we can iterate multiple times.
        segments = list(trancription_segments)
        turns = list(self.diarizied_segments.speaker_diarization.itertracks(yield_label=True))

        if not segments:
            logging.warning("No transcription segments to merge.")
            return

        if not turns:
            logging.warning("No diarization turns to merge.")
            return

        tolerance = 0.25  # seconds
        merged = []
        for turn, _, speaker in turns:
            identified_name = speaker_lookup.get(speaker, speaker)
            merged.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": identified_name,
                "words": []
            })

        # Assign each word to its corresponding diarization window.
        for segment in segments:
            word_list = getattr(segment, "words", []) or []
            for word_info in word_list:
                word_start = word_info.start
                word_text = word_info.word
                for entry in merged:
                    if entry["start"] - tolerance <= word_start <= entry["end"] + tolerance:
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
                f.write(f"[{entry[0]:.4f} - {entry[1]:.4f}] -> {entry[2]}:{entry[3]}\n")

        logging.info(f"Final transcript for-{audio_file} saved to final_transcript_{str(file_counter)}.txt")
        file_counter += 1
    
    def unload_model(self):
        '''
        Unload the ASR and diarisation models from memory to free up resources.
        '''
        
        self.transcribe_model = None
        self.diarisation_model = None
        self.vad_model = None
        self.encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run(self, audio_file = None):
        '''
        Run the ASR model to transcribe and diarise the audio file, and identify speakers.
        Args:
            audio_file (str): Path to the audio file to process.
        '''
        self.diarise_audio(audio_file=audio_file)
        speaker_mapping = self.compare_embeddings(audio_file=audio_file)
        segments = self.transcribe_audio(vad_filter=True,audio_file=audio_file,  word_timestamps=True, batch_size=8)
        
        self.merge_transcriptions_and_diarization(trancription_segments=segments, speaker_mapping=speaker_mapping, audio_file=audio_file)
        # self.create_embeddings()
        
    def split_waveform(self, waveform, sample_rate, chunk_sec, overlap_sec):
        chunks = []
        chunk_samples = chunk_sec * sample_rate
        overlap_samples = overlap_sec * sample_rate
        total_samples = waveform.shape[1]

        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk_wave = waveform[:, start:end]
            offset_time = start / sample_rate
            chunks.append((chunk_wave, offset_time))
            start = start + chunk_samples - overlap_samples

        return chunks   
if __name__ == "__main__":
    transcriber = ASR(model_size=HIN2HINGLISH, device="cuda", compute_type="int8_float16")
    transcriber.run(audio_file="cleaned_audio/tester09_final.wav")