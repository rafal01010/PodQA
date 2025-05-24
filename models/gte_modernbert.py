import torch
from sentence_transformers import SentenceTransformer

class GteModernbert:
    _instance = None
    
    def __new__(cls, model_path=None, device=None):
        if cls._instance is None:
            cls._instance = super(GteModernbert, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, model_path, device=None):
        if not self.initialized:
            if device is None:
                self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            print(f"Loading embedding model: {model_path} on {self.device.upper()}")
            self.model = SentenceTransformer(model_path).to(self.device)
            self.initialized = True
    
    def encode(self, query, convert_to_tensor=True):
        query_embedding = self.model.encode(query, device=self.device, convert_to_tensor=convert_to_tensor)
        if convert_to_tensor:
            return query_embedding.cpu().numpy()
        return query_embedding