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


SMALL = "small"
MEDIUM = "medium"
LARGE = "large-v3"
TURBO = "turbo"
MIN_TIME_THRESHOLD = 0.2
VOICE_DATABASE = "voice_database.db"
SIMILARITY_THRESHOLD = 0.1
UPDATE_THRESHOLD = 0.9
CREATE_NEW_EMBEDDING_THRESHOLD = 0.1

class ASR:
    '''
    Automatic Speech Recognition (ASR) model that transcribes and diarises audio files, and identifies speakers using voice embeddings.
    '''
    def __init__(self, model_size = TURBO, device = "cuda", compute_type = "int8_float16", audio_file=None):
        '''
        Initialize the ASR model with specified parameters.
        Args:
            model_size (str): Size of the Whisper model to use. Options are "small", "medium", "large-v3", "turbo". Default is "turbo".
            device (str): Device to run the model on. Options are "cuda" or "cpu". Default is "cuda".
            compute_type (str): Type of computation to use. Options are "int8_float16
            audio_file (str): Path to the audio file to transcribe and diarize. Default is None.
        '''
        self.HFToken = self.load_HFToken()
        self.transcribe_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.diarisation_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.HFToken)
        self.vad_model = load_silero_vad()
        self.encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device":'cuda'})
        self.audio_file = audio_file
        self.diarizied_segments = None
        # self.diarisation_model2 = SpeakerDiarization()

    @staticmethod
    def load_HFToken():
        '''
        Load the Hugging Face Token form HFToken.txt
        '''
        with open("HFToken.txt" ,"r") as f:
            return f.read().strip("\n")

    def transcribe_audio(self, vad_filter = False):
        '''

        '''
        segments, _ = self.transcribe_model.transcribe(self.audio_file, vad_filter=vad_filter)
        # for segment in segments:
        #     print(f"text: {segment.text}, start: {segment.start}, end: {segment.end}") 

        #test code
        with open("transcriptions.txt", "w") as f:
            for segment in segments:
                f.write(f"text: {segment.text}, start: {segment.start}, end: {segment.end}\n")        
        return segments
    
    def diarise_audio(self):
        '''
        Diarise the audio file using VAD and pyannote.audio's diarisation model.
        '''
        silero_segmentation = Annotation()
        waveform, sample_rate = torchaudio.load(self.audio_file)
        
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
            print(f"start_sec: {start_sec}, end_sec: {end_sec}")
        
        
        self.diarisation_model.to(torch.device("cuda"))
        self.diarizied_segments = self.diarisation_model(audio_data)
        
        with open("diarizied_segments.txt", "w") as f:
            for turn, _, speaker in self.diarizied_segments.itertracks(yield_label=True):
                f.write(f"start={turn.start:.2f}s stop={turn.end:.2f}s speaker={speaker}\n")
    
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
    
    def compare_embeddings(self):
        '''
        Compare the embeddings of the diarised segments with the embeddings stored in the database.
        Returns a mapping of speaker labels to identified names.
        '''
        diarized_segments = self.diarizied_segments
        audio_file = self.audio_file
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
        print(speaker_tensors)
        
        #using SpeechBrain's embedding model to crate an embedding from audio tensor
        with self.connect_db() as conn:
            cursor = conn.cursor()
            for speaker, tensor in speaker_tensors.items():
                highest_similiarity_score = 0
                best_name = "UNKNOWN"
                embedding= self.encoder.encode_batch(tensor)
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
                    speaker_mapping[speaker] = best_name
                else:
                    # print(f"No matching speaker found. best name: {best_name} Highest Score: {highest_similiarity_score:.4f}")
                    speaker_mapping[speaker] = f"UNKNOWN_{best_name}"
           
        # If no speakers were identified, return an empty mapping
        if not speaker_mapping:
            print("\nCould not identify any speakers.")
            return
        print("\n--- Final Speaker Timeline ---")
        # Now, iterate through the original segments and use the map to show the final result
        for turn, _, speaker_label in self.diarizied_segments.itertracks(yield_label=True):
            # Look up the identified name from our map
            identified_name = speaker_mapping.get(speaker_label, speaker_label) # Fallback to the label if not found
            print(f"[{turn.end:.5f} - {turn.start:.5f}] {identified_name}")
                    
    def create_embeddings(self):
        '''
        Create embeddings for the entire audio file and store it in the database.
        '''
        buffer = io.BytesIO()
        waveform, samplerate = torchaudio.load(self.audio_file)
        with self.connect_db() as conn:
            cursor = conn.cursor()
            embedding= self.encoder.encode_batch(waveform)
            torch.save(embedding, buffer)
            embedding_blob = buffer.getvalue()
            embedding_hash = hashlib.sha256(embedding_blob).hexdigest()
            cursor.execute("INSERT INTO VOICE_EMBEDDINGS (name, embedding, embedding_hash) VALUES (?, ?, ?)", ("", embedding_blob, embedding_hash))
            conn.commit()
                              
    def run(self):
        '''
        Run the ASR model to transcribe and diarise the audio file, and identify speakers.
        '''
        self.diarise_audio()
        self.compare_embeddings()
        # self.create_embeddings()
        

transcriber = ASR(model_size=TURBO,audio_file="")
transcriber.run()
