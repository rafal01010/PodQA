import torch
from models.standard_embedding_model import StandardEmbeddingModel
from models.colbert_embedding_model import ColBERTEmbeddingModel

class EmbeddingModelFactory:
    @staticmethod
    def get_model(config_type, config):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")
        
        if config_type == "standard":
            return StandardEmbeddingModel(config, device)
        elif config_type == "modern_colbert":
            return ColBERTEmbeddingModel(config, device)
        else:
            raise ValueError(f"Unknown embedding type: {config_type}")