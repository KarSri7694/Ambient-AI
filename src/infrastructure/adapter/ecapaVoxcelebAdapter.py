from core.models import SpeakerEmbedding, SpeakerMapping
from application.ports.identity_port import SpeakerIdentityPort
from application.ports.voice_repository_port import VoiceRepository
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import SpeakerRecognition, EncoderClassifier
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EcapaVoxcelebAdapter(SpeakerIdentityPort):
    def __init__(self,voice_repo = VoiceRepository, model_name_or_path: str = "spkrec-ecapa-voxceleb/"):
        self.voice_repo = voice_repo
        self.classifier = EncoderClassifier.from_hparams(source=model_name_or_path, savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})

    def identify_speaker(self, audio_tensor: torch.Tensor, original_label: str, threshold: float = 0.7) -> str:
        """
        Identify the speaker from the given audio metadata using ECAPA-TDNN model.
        Args:
            audio_metadata (SpeakerEmbedding): Audio metadata containing the embedding tensor.
        Returns:
            SpeakerMapping: object containg identified label, original label and score
        """
        # Perform speaker identification
        embedding = self.classifier.encode_batch(audio_tensor)
        embedding = embedding.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        known_embeddings = self.voice_repo.get_all_embeddings()
        
        best_match = "UNKNOWN"
        highest_score = threshold
        for speaker_label, known_embedding in known_embeddings.items():
            known_embedding = known_embedding.to(embedding.device)
            score = F.cosine_similarity(embedding, known_embedding, dim= 2).item()
            if score > highest_score:
                highest_score = score
                best_match = speaker_label
        return SpeakerMapping(
            original_label=original_label,
            identified_label=best_match,
            score = highest_score
        )
        
    def create_speaker_embedding(self, audio_tensor: torch.Tensor, speaker_label: str) :
        """
        Create a speaker embedding from the given audio tensor using ECAPA-TDNN model.
        Args:
            audio_tensor (torch.Tensor): Audio tensor to create embedding from.
            speaker_label (str): Label of the speaker.
        Returns:
            bool: True if embedding creation and storage was successful, False otherwise.
        """
        # Generate speaker embedding
        embedding = self.classifier.encode_batch(audio_tensor)
        speaker_embedding = SpeakerEmbedding(
            speaker_label=speaker_label,
            embedding=embedding
        )
        # Store the embedding in the voice repository
        try:
            self.voice_repo.store_embedding(speaker_embedding)
            return True
        except Exception as e:
            logging.error(f"Error storing embedding: {e}")
            return False
        
    def unload_model(self):
        """
        Unloads the ECAPA-TDNN model from memory.
        """
        self.classifier = None