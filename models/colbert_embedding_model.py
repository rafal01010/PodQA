import torch
import pyarrow as pa
from transformers import AutoModel, AutoTokenizer
from srt_embeddings.utils import generate_id

class ColBERTEmbeddingModel:
    def __init__(self, config, device):
        self.model_path = config["model_path"]
        self.embedding_dimension = config["embedding_dimension"]
        self.doc_length = config["doc_length"]
        self.query_length = config["query_length"]
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(device)
        self.batch_size = 8  # Smaller batch size for ColBERT due to token-level embeddings
    
    def encode(self, texts):
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize with padding and truncation
            inputs = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.doc_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get the token embeddings (last hidden state)
                token_embeddings = outputs.last_hidden_state
                
                # Normalize embeddings
                token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=2)
                
                # Convert to numpy and store
                # We store the whole token-level representation as a flattened vector
                for embedding in token_embeddings:
                    # Convert to list and add to results
                    all_embeddings.append(embedding.cpu().flatten().numpy().tolist())
            
            print(f"Processed batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
        
        return all_embeddings
    
    def get_schema(self):
        # For ColBERT, we store the flattened token-level embeddings
        # The dimension is embedding_dimension * doc_length because we store all token embeddings
        total_dim = self.embedding_dimension * self.doc_length
        
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("video_url", pa.string()),
            pa.field("text", pa.string()),
            pa.field("start_time", pa.string()),
            pa.field("end_time", pa.string()),
            pa.field("start_seconds", pa.float64()),
            pa.field("end_seconds", pa.float64()),
            pa.field("colbert_embedding", pa.list_(pa.float32(), total_dim))
        ])
    
    def prepare_records(self, file_name, entries, embeddings, video_url):
        records = []
        for entry, embedding in zip(entries, embeddings):
            record_id = generate_id(file_name, entry['start_seconds'], entry['end_seconds'], entry['text'])
            records.append({
                'id': record_id,
                'file_name': file_name,
                'video_url': video_url,
                'text': entry['text'],
                'start_time': entry['start_time'],
                'end_time': entry['end_time'],
                'start_seconds': entry['start_seconds'],
                'end_seconds': entry['end_seconds'],
                'colbert_embedding': embedding
            })
        return records