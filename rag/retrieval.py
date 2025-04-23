import pandas as pd

def retrieve_context(query, embedding_model, reranker, table, overfetch_multiplier=3, chunks_per_doc=3, max_docs=5):
    # query = query.split("Current question:")[1]
    # print("RETRIEVAL QUERY")
    # print(query)
    query_embedding = embedding_model.encode(query)

    top_k = max_docs * chunks_per_doc * overfetch_multiplier
    results = table.search(query_embedding, vector_column_name="embedding").limit(top_k).to_pandas()
    
    if results.empty:
        return []
    
    
    reranked = reranker.rerank(query, results)
    grouped = reranked.groupby('file_name')
    
    processed_groups = []
    for name, group in grouped:
        sorted_group = group.sort_values(by='_rerank_score', ascending=False)
        top_chunks = sorted_group.head(chunks_per_doc)
        if not top_chunks.empty:
            max_score = top_chunks['_rerank_score'].iloc[0]
            processed_groups.append((max_score, name, top_chunks))
    
    processed_groups.sort(key=lambda x: x[0], reverse=True)
    
    selected_groups = processed_groups[:max_docs]
    final_df = pd.concat([g[2] for g in selected_groups]).sort_values(by='_rerank_score', ascending=False)
    
    return [
        {
            "text": row.text,
            "file_name": row.file_name,
            "video_url": row.video_url,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "start_seconds": row.start_seconds,
            "end_seconds": row.end_seconds,
            "score": row._rerank_score
        }
        for _, row in final_df.iterrows()
    ]