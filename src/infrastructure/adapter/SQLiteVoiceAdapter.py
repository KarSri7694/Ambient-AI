from core.models import SpeakerEmbedding, TranscriptionResult, DiarizationResult
from application.ports.voice_repository_port import VoiceRepository
import sqlite3
import torch
import hashlib
import io

class SQLiteVoiceAdapter(VoiceRepository):
    def __init__(self, db_path: str):
        """
        Initialize the SQLiteVoiceAdapter with a connection to the SQLite database.
        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS VOICE_EMBEDDINGS (
                            uid INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            embedding BLOB NOT NULL,
                            embedding_hash TEXT NOT NULL UNIQUE,
                            audio BLOB
                    )''')
        self.connection.commit()
    
    def store_embedding(self, speaker_embedding: SpeakerEmbedding) -> None:
        """
        Store the given speaker embedding in the repository.
        Args:
            speaker_embedding (SpeakerEmbedding): The speaker embedding to store.
        """
        speaker_label , embedding = speaker_embedding.speaker_label, speaker_embedding.embedding
        buffer = io.BytesIO()
        torch.save(embedding, buffer)
        embedding_blob = buffer.getvalue()
        embedding_hash = hashlib.sha256(embedding_blob).hexdigest()
        self.cursor.execute(
                            "INSERT INTO VOICE_EMBEDDINGS (name, embedding, embedding_hash) VALUES (?, ?, ?)", (speaker_label, embedding_blob, embedding_hash))
        self.connection.commit()
        
        
    def get_all_embeddings(self) -> dict[str, list[float]]:
        """Retrieve all speaker embeddings from the repository."""
        self.cursor.execute('SELECT name, embedding FROM VOICE_EMBEDDINGS')
        rows = self.cursor.fetchall()
        embeddings = {}
        for row in rows:
            speaker_label = row[0]
            embedding_blob = row[1]
            buffer = io.BytesIO(embedding_blob)
            embedding = torch.load(buffer)
            embedding = embedding.squeeze(0).squeeze(0)
            embeddings[speaker_label] = embedding
        return embeddings
    
