import logging

from application.ports.LLMProvider import LLMProvider
from application.services.llm_interaction_service import LLMInteractionService
from utils.kv_state_handling import KVStateControl



class CheckpointControlService:
    def __init__(self, llm_service: LLMInteractionService, llm_provider: LLMProvider):
        self.llm_service = llm_service
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(__name__)
    