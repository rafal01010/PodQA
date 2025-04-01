import time

def rag_pipeline(
    query,
    retriever_function,
    generator,
    embedding_model,
    table,
    top_k=5,
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
        top_k: Number of contexts to retrieve
        **generator_kwargs: Additional keyword arguments for the generator
        
    Returns:
        Generated response
    """
    retrieved_contexts = retriever_function(query, embedding_model, table, top_k=top_k)

    start_time = time.time()
    print("gen start")
    response = generator.generate(
        query=query,
        retrieved_contexts=retrieved_contexts,
        **generator_kwargs
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for gen: {elapsed_time} seconds")

    
    return response