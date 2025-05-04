import pyarrow as pa
from sentence_transformers import SentenceTransformer
from srt_embeddings.utils import generate_id

class StandardEmbeddingModel:
    def __init__(self, config, device):
        self.model_path = config["model_path"]
        self.embedding_dimension = config["embedding_dimension"]
        self.device = device
        self.model = SentenceTransformer(self.model_path).to(device)
        self.batch_size = 16
    
    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            device=self.device,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy().tolist()
    
    def get_schema(self):
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("video_url", pa.string()),
            pa.field("text", pa.string()),
            pa.field("start_time", pa.string()),
            pa.field("end_time", pa.string()),
            pa.field("start_seconds", pa.float64()),
            pa.field("end_seconds", pa.float64()),
            pa.field("embedding", pa.list_(pa.float32(), self.embedding_dimension))
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
                'embedding': embedding
            })
        return records