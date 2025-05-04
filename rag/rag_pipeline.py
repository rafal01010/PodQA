import time

def rag_pipeline(
    query,
    retriever_function,
    generator,
    embedding_model,
    reranker,
    table,
    conversation_history=None,
    **generator_kwargs
):
    """
    Complete RAG pipeline combining retrieval and generation.
    
    Args:
        query: Query string used
        retriever_function: Function to retrieve relevant contexts
        generator: Initialized Gemma3Generator instance
        embedding_model: The embedding model
        reranker: The reranker model
        table: The database table
        conversation_history: Previous conversation messages
        **generator_kwargs: Additional keyword arguments for the generator
        
    Returns:
        Dictionary containing response and source information
    """
    start_time = time.time()
    retrieved_contexts = retriever_function(query, embedding_model, reranker, table)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for retrieval: {elapsed_time} seconds")

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
        
        video_id = context['video_url'].split('v=')[1]
        timestamp_url = f"https://youtu.be/{video_id}?t={int(context['start_seconds'])}"
        
        result['sources'][file_name]['url'].append(timestamp_url)
        result['sources'][file_name]['text'].append(context['text'])
    
    return result