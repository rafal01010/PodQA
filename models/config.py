CONFIG = {
    "embedding_type": "standard",  
    "databases": {
        "standard": {
            "path": "transcripts_lancedb",
            "model_path": "/Users/dave/AI/models/gte-modernbert-base",
            "embedding_dimension": 768
        },
        "modern_colbert": {
            "path": "transcripts_moderncolbert_lancedb",
            "model_path": "lightonai/GTE-ModernColBERT-v1",
            "embedding_dimension": 128,
            "doc_length": 300,
            "query_length": 32
        }
    }
}