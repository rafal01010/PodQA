import lancedb
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")
torch_dtype = torch.bfloat16 if torch.backends.mps.is_available() or torch.cuda.is_available()  else torch.float32

db = lancedb.connect("./srt_embeddings/transcripts_lancedb")
table = db.open_table("transcripts")

embed_model_path = "/Users/dave/AI/models/gte-modernbert-base"
embedding_model = SentenceTransformer(embed_model_path).to(device)

def retrieve_context(query, embedding_model, table, top_k=5):
    query_embedding = embedding_model.encode(query, device=device, convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    
    results = table.search(query_embedding, vector_column_name="embedding").limit(top_k).to_pandas()
    
    return [
        {
            "text": row.text,
            "file_name": row.file_name,
            "video_url": row.video_url,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "start_seconds": row.start_seconds,
            "end_seconds": row.end_seconds,
            "score": row._distance
        }
        for _, row in results.iterrows()
    ]



def rag_pipeline(query):
    context = retrieve_context(query, embedding_model, table)
    for item in context:
        print("SAMPLE")
        print(item)


user_in = input("Enter prompt: ")
rag_pipeline(user_in)



