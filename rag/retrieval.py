import pandas as pd

def retrieve_context(query, embedding_model, table, top_k=20, chunks_per_doc=1, max_docs=3):
    query_embedding = embedding_model.encode(query)
    
    results = table.search(query_embedding, vector_column_name="embedding").limit(top_k).to_pandas()
    
    if results.empty:
        return []
    
    grouped = results.groupby('file_name')
    
    processed_groups = []
    for name, group in grouped:
        sorted_group = group.sort_values(by='_distance')
        top_chunks = sorted_group.head(chunks_per_doc)
        if not top_chunks.empty:
            min_distance = top_chunks['_distance'].iloc[0]
            processed_groups.append((min_distance, name, top_chunks))
    
    processed_groups.sort(key=lambda x: x[0])
    
    selected_groups = processed_groups[:max_docs]

    final_df = pd.concat([g[2] for g in selected_groups]).sort_values(by='_distance')
    
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
        for _, row in final_df.iterrows()
    ]