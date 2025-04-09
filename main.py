import torch
import lancedb
from models.gte_modernbert import GteModernbert
from models.gemma import Gemma3
from rag.retrieval import retrieve_context
from rag.rag_pipeline import rag_pipeline

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

db = lancedb.connect("./srt_embeddings/transcripts_lancedb")
table = db.open_table("transcripts")

embed_model_path = "/Users/dave/AI/models/gte-modernbert-base"
embedding_model = GteModernbert(embed_model_path)

llm_path = "/Users/dave/AI/models/gemma-3-4b-it"
generator = Gemma3(model_path=llm_path)

for _ in range(1):
    user_in = input("Enter prompt: ")

    result = rag_pipeline(
        query=user_in,
        retriever_function=retrieve_context,
        generator=generator,
        embedding_model=embedding_model,
        table=table,
        top_k=20,
        temperature=0.3,
        max_new_tokens=500,
        do_sample=True
    )

    # print("\nGenerated Response:")
    # print(result['response'])
    # print(result['sources'])

    