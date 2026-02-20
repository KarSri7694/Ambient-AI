from abc import ABC, abstractmethod

class ModelManager(ABC):
    
    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load a model from the specified path."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the currently loaded model to free up resources."""
        pass
    
    @abstractmethod
    def get_current_model(self) -> str:
        """Get the name of the currently loaded model."""
        pass