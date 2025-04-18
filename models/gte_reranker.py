import torch
from sentence_transformers import CrossEncoder

class GteReranker:
    _instance = None

    def __new__(cls, model_path, device=None):
        if cls._instance is None:
            cls._instance = super(GteReranker, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_path, device=None):
        if self.initialized:
            return

        if device is None:
            self.device = (
                "mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )
        else:
            self.device = device

        print(f"Loading reranker model: {model_path} on {self.device.upper()}")
        self.model = CrossEncoder(model_path, device=self.device)
        self.initialized = True

    def rerank(self, query, results_df):
        if results_df.empty:
            return results_df
        
        pairs = [(query, row['text']) for _, row in results_df.iterrows()]
        scores = self.model.predict(pairs)
        results_df['_rerank_score'] = scores

        ranked_df = results_df.sort_values(by='_rerank_score', ascending=False)
        
        return ranked_df