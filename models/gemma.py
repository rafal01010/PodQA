from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from typing import List, Dict, Any, Optional
import torch
import threading

class Gemma3:
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, model_path=None, device=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Gemma3, cls).__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: str = "google/gemma-3-8b",
        device: str = None
    ):
        """
        Initialize the Gemma 3 generator with model and processor.
        
        Args:
            model_name: Hugging Face model identifier for Gemma 3
            device: Device to run the model on ("cuda", "cpu", or "mps").
                   If None, will auto-detect the best available device.
        """
        if self.initialized:
            return

        with self.__class__._lock:
            if self.initialized:
                return

            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            self.device = device
            
            self.use_bfloat16 = device in ["cuda", "mps"] and hasattr(torch, "bfloat16")
            self.dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
            
            print(f"Loading Gemma 3 model: {model_path} on {device} with {self.dtype} precision")
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(model_path)
            
            self.model = self.model.to(device=self.device, dtype=self.dtype)
                
            print(f"Model loaded successfully on {device}")
            
            self.initialized = True
        
    def generate(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        conversation_history: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 100,
        system_prompt: Optional[str] = None,
        do_sample: bool = False
    ) -> str:
        """
        Generate a response using Gemma 3 based on the query, conversation history and retrieved contexts.
        
        Args:
            query: User query string
            retrieved_contexts: List of context dictionaries retrieved from the database
            conversation_history: Optional string containing conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the model behavior
            do_sample: Whether to use sampling for generation
            
        Returns:
            Generated response as a string
        """
        context_texts = [f"Transcript taken from episode titled: {context['file_name'].split('｜')[0]}\n{context['text']}" for context in retrieved_contexts if "text" in context]
        combined_context = "\n\n---\n\n".join(context_texts)
        
        if system_prompt is None:
            system_prompt = (
                """
                You are an assistant specialized in answering questions about the Trash Taste podcast based on transcription data.

                IMPORTANT: NEVER mention, reference, or acknowledge any "context" or "transcripts" in your responses. Users cannot see this context and will be confused by such references.

                The Trash Taste podcast features hosts Garnt (Gigguk), Joey (The Anime Man), and Connor (CDawgVA) discussing anime, Japanese culture, and their personal experiences living in Japan.

                When responding to queries:
                - Answer questions accurately based on the context provided
                - The provided context are 1 minute transcripts from the podcast episodes
                - For each context, the first line contains the title of the episode on where the transcript or information is from
                - For episodes with guests the first or the title will also contain the name of the guest usually in parentheses (ft. name of guest)
                - If the question is about finding a specific episode you can use the first line of the context because it contains the title of the episode on where the transcript is from
                - NEVER use phrases like "based on the context" or "according to the transcripts"
                - NEVER cite sources or reference specific episodes/timestamps
                - Do NOT mention that you are answering based on the context
                - Do NOT mention that you are using transcripts as context
                - Do NOT cite sources or reference specific episodes/timestamps in your responses
                - Do NOT include the provided context in your response
                - DO use the context to inform your answers but present information in a conversational manner
                - If the context doesn't contain sufficient information to answer a question, clearly state that you don't have enough information rather than making up details
                - If asked about topics not related to Trash Taste, politely redirect to podcast-related content
                - When asked to elaborate or provide more details, ensure you draw from different portions of the context not already used in previous responses
                - Avoid repeating the same information when users ask for more details
                - Maintain awareness of what information you've already shared in the conversation
                - If asked to elaborate and you've already shared all relevant information from the context, clearly state "I've shared all the information I have about this topic" rather than repeating yourself
                - Track the depth of information provided to ensure progressive disclosure when users ask for more details
                - For follow-up questions, prioritize information not yet shared from the context
                - Distinguish between different levels of detail when responding to initial questions versus requests for elaboration
                - Structure your knowledge to provide basic information first, then more specific details upon request    

                Your goal is to help users access information from the podcast naturally, as if they were having a conversation with someone who has comprehensive knowledge of all Trash Taste episodes.
                """
            )
        
        user_message = f"Answer this question about Trash Taste: {query}"
        if conversation_history:
            user_message = f"Conversation history:\n{conversation_history}\n\nAnswer the current question about Trash Taste: {query}"
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"Use this information (but never reference it):\n\n{combined_context}\n\n{user_message}"
                    }
                ]
            },
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                top_k=40 if do_sample else 0,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]
        
        response = self.processor.decode(generation, skip_special_tokens=True)
        
        return response
    
    def rewrite_query(
        self,
        current_query: str,
        conversation_history: str,
        temperature: float = 0.0,
        max_new_tokens: int = 100,
        do_sample: bool = False
    ) -> str:
        """
        Rewrite an ambiguous or context-dependent query into a self-contained query
        using the Gemma 3 model, based on conversation history string.
        
        Args:
            current_query: The latest user query that may contain contextual references
            conversation_history: String containing the conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate for the rewritten query
            do_sample: Whether to use sampling for generation
            
        Returns:
            A rewritten query that is self-contained, or the original query if not context-dependent
        """
        system_prompt = """
        You are an AI assistant whose ONLY task is to decide whether the user's *current* query is context-dependent and, if so, rewrite it into a fully self-contained version. **Err on the side of rewriting**—when in doubt, expand the query.
        
        **Rewriting rules**
        1. If the current query depends on previous messages (e.g. pronouns like “he”, “it”, “them”; adverbs like “here”, “there”; elliptical phrases like “tell me more”, “what about the second one?”), rewrite the query so that every reference becomes explicit. Keep the original wording and tone as much as possible.
        2. If—and only if—the current query is **already fully self-contained**, return it unchanged.
        3. Output **exactly one line** containing **only** the final query—no prefixes, quotes, or commentary.
        
        **Examples**
        ### Example 1 (should rewrite)
        Conversation history:
        User 1: Describe Chris Broad.
        User 2: He is a British YouTuber known as Abroad in Japan.
        Current query: Describe him more
        → Output:
        Describe Chris Broad more
        
        ### Example 2 (should rewrite)
        Conversation history:
        User 1: Tell me about the Trash Taste hosts.
        Current query: And Connor?
        → Output:
        Tell me about Connor from Trash Taste
        
        ### Example 3 (already self-contained, keep)
        Conversation history:
        User 1: What is the capital of France?
        Current query: What is the population of Paris?
        → Output:
        What is the population of Paris?
        
        ### Example 4 (should rewrite)
        Conversation history:
        User 1: When did Cyberpunk 2077 come out?
        Current query: What about the DLC?
        → Output:
        When did the DLC for Cyberpunk 2077 come out?
        
        Follow the rules and examples **exactly**.
        """
        
        user_message = f"""
        Conversation history:
        {conversation_history}
        
        Current query: "{current_query}"
        
        Rewritten query:
        """
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                top_k=40 if do_sample else 0,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]
        
        rewritten_query = self.processor.decode(generation, skip_special_tokens=True)
        rewritten_query = rewritten_query.strip()
        
        return rewritten_query
