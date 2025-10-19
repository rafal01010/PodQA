import time
import pandas as pd
from pylate import models
import torch
from models.gte_modernbert import GteModernbert
from models.gte_moderncolbert import GteModernColbert
from urllib.parse import urlparse, parse_qs


def _extract_youtube_id(url: str):
    """
    Extract a YouTube video identifier from a variety of supported URL formats.
    
    Falls back to None when the URL cannot be parsed, which lets callers
    decide how to handle missing identifiers without raising.
    """
    if not url:
        return None

    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    netloc = (parsed.netloc or "").lower()

    # Standard watch URLs where the id lives in the "v" query parameter.
    query_params = parse_qs(parsed.query)
    if query_params.get("v"):
        return query_params["v"][0]

    path = (parsed.path or "").lstrip("/")

    # Short links such as youtu.be/<id>
    if netloc.endswith("youtu.be") and path:
        return path.split("/")[0]

    # Embedded player or shorts URLs keep the id in the first path segment.
    youtube_hosts = ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com")
    if any(netloc.endswith(host) for host in youtube_hosts) and path:
        for prefix in ("embed/", "shorts/"):
            if path.startswith(prefix):
                remainder = path[len(prefix):]
                if remainder:
                    return remainder.split("/")[0]

    return None

def hybrid_retrieve_context(query, embedding_model_path, table, overfetch_multiplier=3, chunks_per_doc=3, max_docs=5):
    embedding_model = GteModernbert(embedding_model_path)
    query_embedding = embedding_model.encode(query)

    top_k = max_docs * chunks_per_doc * overfetch_multiplier
    results = table.search(query_type="hybrid", vector_column_name="embedding", fts_columns="text").vector(query_embedding).text(query).limit(top_k).to_pandas()

    if results.empty:
        return []

    grouped = results.groupby('file_name')
    
    processed_groups = []
    for name, group in grouped:
        sorted_group = group.sort_values(by='_relevance_score', ascending=False)
        top_chunks = sorted_group.head(chunks_per_doc)
        if not top_chunks.empty:
            max_score = top_chunks['_relevance_score'].iloc[0]
            processed_groups.append((max_score, name, top_chunks))
    
    processed_groups.sort(key=lambda x: x[0], reverse=True)
    
    selected_groups = processed_groups[:max_docs]
    final_df = pd.concat([g[2] for g in selected_groups]).sort_values(by='_relevance_score', ascending=False)
    
    return [
        {
            "text": row.text,
            "file_name": row.file_name,
            "video_url": row.video_url,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "start_seconds": row.start_seconds,
            "end_seconds": row.end_seconds,
            "score": row._relevance_score
        }
        for _, row in final_df.iterrows()
    ]

def colbert_retrieve_context(query, modern_colbert_path, table, overfetch_multiplier=3, chunks_per_doc=3, max_docs=5):
    print("loading model")
    embedding_model = GteModernColbert(modern_colbert_path)
    print("model loaded")

    query_embedding = embedding_model.encode(query, True)
    print("query encoded")

    top_k = max_docs * chunks_per_doc * overfetch_multiplier
    results = table.search(query_embedding, vector_column_name="multivector_embedding").limit(top_k).to_pandas()
    print("table retrieved")

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

def rag_pipeline(
    query,
    generator,
    table,
    embedding_model_path=None,
    modern_colbert_path=None,
    retrieve_type="hybrid",
    conversation_history=None,
    **generator_kwargs
):
    """
    Complete RAG pipeline combining retrieval and generation.
    
    Args:
        query: Query string used
        generator: Initialized Gemma3Generator instance
        embedding_model: The embedding model
        modern_colbert_path: The path for ModernColBERT model
        reranker: The reranker model
        table: The database table
        conversation_history: Previous conversation messages
        **generator_kwargs: Additional keyword arguments for the generator
        
    Returns:
        Dictionary containing response and source information
    """
    start_time = time.time()
    if retrieve_type.lower() == "colbert":
        retrieved_contexts = colbert_retrieve_context(query, modern_colbert_path, table)
    else:
        retrieved_contexts = hybrid_retrieve_context(query, embedding_model_path, table)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for retrieval: {elapsed_time} seconds")

    retrieved_contexts = [context for context in retrieved_contexts if context.get('video_url')]

    start_time = time.time()
    response = generator.generate(
        query=query,
        conversation_history=conversation_history,
        retrieved_contexts=retrieved_contexts,
        **generator_kwargs
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for generation: {elapsed_time} seconds")

    result = {
        'response': response,
        'sources': {}
    }
    
    for context in retrieved_contexts:
        file_name = context['file_name']
        
        if file_name not in result['sources']:
            result['sources'][file_name] = {
                'url': [],
                'text': []
            }
        
        video_id = _extract_youtube_id(context['video_url'])

        if video_id:
            start_seconds = context.get('start_seconds') or 0
            timestamp_url = f"https://youtu.be/{video_id}?t={int(start_seconds)}"
        else:
            # Fall back to the original URL when we cannot derive a canonical timestamp URL.
            timestamp_url = context['video_url']
        
        result['sources'][file_name]['url'].append(timestamp_url)
        result['sources'][file_name]['text'].append(context['text'])
    
    return result
