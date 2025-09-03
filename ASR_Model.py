from faster_whisper import WhisperModel
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
# import torchaudio without warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')
    warnings.filterwarnings("ignore", category=FutureWarning, module='torchaudio')    
    import torchaudio

#Whisper model sizes
SMALL = "small"
MEDIUM = "medium"
LARGE = "large-v3"
TURBO = "turbo"

#Model identifiers
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
VAD_MODEL = "silero_vad"
EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

#File paths
VOICE_DATABASE = "voice_database.db"
HF_TOKEN_FILE = "HFToken.txt"

#Thresholds
MIN_TIME_THRESHOLD = 0.2
SIMILARITY_THRESHOLD = 0.1
UPDATE_THRESHOLD = 0.9
CREATE_NEW_EMBEDDING_THRESHOLD = 0.1

#File counter
file_counter = 1

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
        self.transcribe_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.diarisation_model = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=self.HFToken)
        self.vad_model = load_silero_vad()
        self.encoder = EncoderClassifier.from_hparams(source=EMBEDDING_MODEL,run_opts={"device":'cuda'})
        self.diarizied_segments = None

    @staticmethod
    def load_HFToken():
        '''
        Load the Hugging Face Token form HFToken.txt
        '''
        with open("HFToken.txt" ,"r") as f:
            return f.read().strip("\n")

    def transcribe_audio(self, vad_filter = False, task = "transcribe", language="hi" , audio_file=None):
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
        segments, _ = self.transcribe_model.transcribe(audio_file, vad_filter=vad_filter, task=task, language=language)     
        return segments
    
    def diarise_audio(self, audio_file=None):
        '''
        Diarise the audio file using VAD and pyannote.audio's diarisation model.
        Args:
            audio_file (str): Path to the audio file to diarise.
        '''
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
        
    def connect_db(self):
        '''
        Connect to the SQLite database and create the VOICE_EMBEDDINGS table if it doesn't exist.
        '''
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
    
    def compare_embeddings(self, audio_file=None):
        '''
        Compare the embeddings of the diarised segments with the embeddings stored in the database.
        Returns a mapping of speaker labels to identified names.
        Args:
            audio_file (str): Path to the audio file to process.
        Returns:
            speaker_mapping (dict): Mapping of speaker labels to identified names.
        '''
        diarized_segments = self.diarizied_segments
        waveform, samplerate = torchaudio.load(audio_file)
        total_sample = waveform.shape[1]
        speaker_audio_tensor = defaultdict(list)
        
        speaker_mapping = {}
        for turn, _, speaker in diarized_segments.itertracks(yield_label=True):
            segment_duration = turn.end - turn.start
            if segment_duration > MIN_TIME_THRESHOLD:
                start_sample = int(turn.start*samplerate)
                end_sample = int(turn.end*samplerate)
                cropped_tensor = waveform[:, start_sample:end_sample]
                speaker_audio_tensor[speaker].append(cropped_tensor)
        
        speaker_tensors = {
            speaker : torch.cat(tensor_list, dim=1)
            for speaker, tensor_list in speaker_audio_tensor.items()
        }
        
        #using SpeechBrain's embedding model to crate an embedding from audio tensor
        with self.connect_db() as conn:
            cursor = conn.cursor()
            for speaker, tensor in speaker_tensors.items():
                highest_similiarity_score = 0
                best_name = "UNKNOWN"
                embedding= self.encoder.encode_batch(tensor)
                # embedding shape = (1, 1, 192)
                for row in cursor.execute("SELECT name, embedding FROM VOICE_EMBEDDINGS"):
                    name, embedding_blob =row[0], row[1] 
                    buffer = io.BytesIO(embedding_blob)
                    stored_embedding = torch.load(buffer)
                    similiarity_score = torch.nn.functional.cosine_similarity(embedding, stored_embedding, dim=2).item()
                    if similiarity_score > highest_similiarity_score:
                        highest_similiarity_score = similiarity_score
                        best_name = name
                
                if highest_similiarity_score > SIMILARITY_THRESHOLD:
                    # print(f"Speaker identified as: {best_name}, Accuracy: {(highest_similiarity_score)*100:.2f}%")
                    speaker_mapping[speaker] = f"{best_name} ({(highest_similiarity_score)*100:.2f}%)" 
                else:
                    # print(f"No matching speaker found. best name: {best_name} Highest Score: {highest_similiarity_score:.4f}")
                    speaker_mapping[speaker] = f"UNKNOWN_{best_name} ({(highest_similiarity_score)*100:.2f}%)"
           
        # If no speakers were identified, return an empty mapping
        if not speaker_mapping:
            print("\nCould not identify any speakers.")
            return
        return speaker_mapping
                    
    def create_embeddings(self, audio_file=None):
        '''
        Create embeddings for the entire audio file and store it in the database.
        Args:
            audio_file (str): Path to the audio file to process.
        '''
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
            speaker_mapping (dict): Mapping of speaker labels to identified names.
            audio_file (str): Path to the audio file.
        '''
        global file_counter
        
        if self.diarizied_segments is None:
            print("Diarisation has not been performed yet.")
            return
        
        final_transcript = []
        
        for segment in trancription_segments:
            start_time = segment.start
            end_time = segment.end
            text = segment.text
            mid_time = (start_time + end_time) / 2
            for turn, _, speaker in self.diarizied_segments.itertracks(yield_label=True):
                identified_name = speaker_mapping.get(speaker, speaker)
                if turn.start <= mid_time <= turn.end:
                    # final transcript is a list of tuples (start_time, end_time, speaker, text)
                    final_transcript.append((start_time, end_time, identified_name, text))
                    break
                    
        with open(f"final_transcript_{str(file_counter)}.txt", "w", encoding="utf-8") as f:
            for entry in final_transcript:
                f.write(f"[{entry[0]:.4f} - {entry[1]:.4f}] -> {entry[2]}:{entry[3]}\n")
            
            print(f"\nFinal transcript for-{audio_file} saved to final_transcript_{str(file_counter)}.txt")
            file_counter += 1
    
    def run(self, audio_file = None):
        '''
        Run the ASR model to transcribe and diarise the audio file, and identify speakers.
        Args:
            audio_file (str): Path to the audio file to process.
        '''
        self.diarise_audio(audio_file=audio_file)
        speaker_mapping = self.compare_embeddings(audio_file=audio_file)
        segments = self.transcribe_audio(vad_filter=True,audio_file=audio_file ,language="hi")
        self.merge_transcriptions_and_diarization(trancription_segments=segments, speaker_mapping=speaker_mapping, audio_file=audio_file)
        # self.create_embeddings()
        
transcriber = ASR(model_size=TURBO, device="cuda", compute_type="int8_float16")
transcriber.run(audio_file="")
