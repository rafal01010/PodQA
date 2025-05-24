import torch
from pylate import models

class GteModernColbert:
    _instance = None
    
    def __new__(cls, model_path=None, device=None):
        if cls._instance is None:
            cls._instance = super(GteModernColbert, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, model_path, device=None):
        if not self.initialized:
            if device is None:
                self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            print(f"Loading ModernColBERT model: {model_path} on {self.device.upper()}")
            self.model = models.ColBERT(
                model_name_or_path=model_path,
                document_length=1500,
                device=device
            )
            self.initialized = True
    
    def encode(self, query):
        multivector_embeddings = self.model.encode(
            query,
            is_query=False,
            show_progress_bar=False,
        )
        return multivector_embeddings.tolist()