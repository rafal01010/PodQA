import time

def rag_pipeline(
    query,
    retriever_function,
    generator,
    embedding_model,
    reranker,
    table,
    **generator_kwargs
):
    """
    Complete RAG pipeline combining retrieval and generation.
    
    Args:
        query: User query string
        retriever_function: Function to retrieve relevant contexts
        generator: Initialized Gemma3Generator instance
        embedding_model: The embedding model
        table: The database table
        **generator_kwargs: Additional keyword arguments for the generator
        
    Returns:
        Generated response
    """
    retrieved_contexts = retriever_function(query, embedding_model, reranker, table,)

    print(query)

    start_time = time.time()
    response = generator.generate(
        query=query,
        retrieved_contexts=retrieved_contexts,
        **generator_kwargs
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for gen: {elapsed_time} seconds")

    result = {}
    result['response'] = response
    
    temp = {}
    for context in retrieved_contexts:
        if context['file_name'] not in temp:
            temp[context['file_name']] = {}
            temp[context['file_name']]['url'] = []
            temp[context['file_name']]['text'] = []
        video_id = context['video_url'].split('v=')[1]
        temp[context['file_name']]['url'].append(f"https://youtu.be/{video_id}?t={int(context['start_seconds'])}")
        temp[context['file_name']]['text'].append(context['text'])
    
    result['sources'] = temp
    
    return result