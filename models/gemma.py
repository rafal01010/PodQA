from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from typing import List, Dict, Any, Optional
import torch

class Gemma3:
    _instance = None
    
    def __new__(cls, model_path=None, device=None):
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
        if not self.initialized:
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
        temperature: float = 0.0,
        max_new_tokens: int = 100,
        system_prompt: Optional[str] = None,
        do_sample: bool = False
    ) -> str:
        """
        Generate a response using Gemma 3 based on the query and retrieved contexts.
        
        Args:
            query: User query string
            retrieved_contexts: List of context dictionaries retrieved from the database
            temperature: Sampling temperature (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the model behavior
            do_sample: Whether to use sampling for generation
            
        Returns:
            Generated response as a string
        """
        context_texts = [f"Transcript taken from episode titled: {context['file_name'].split('｜')[0]}\n{context['text']}" for context in retrieved_contexts if "text" in context]
        combined_context = "\n\n---\n\n".join(context_texts)
        print(combined_context)
        
        if system_prompt is None:
            
            system_prompt = (
                """
                You are an assistant specialized in answering questions about the Trash Taste podcast based on transcription data. 

                The Trash Taste podcast features hosts Garnt (Gigguk), Joey (The Anime Man), and Connor (CDawgVA) discussing anime, Japanese culture, and their personal experiences living in Japan.

                When responding to queries:
                - Answer questions accurately based on the context provided
                - The provided context are 1 minute transcripts from the podcast episodes
                - Try to use multiple contexts when generating a response but if the other contexts is irrelevant then do NOT force using other contexts
                - Do NOT mention that you are answering based on the context
                - Do NOT mention that you are using transcripts as context
                - Do NOT cite sources or reference specific episodes/timestamps in your responses
                - Do NOT include the provided context in your response
                - DO use the context to inform your answers but present information in a conversational manner
                - If the context doesn't contain sufficient information to answer a question, clearly state that you don't have enough information rather than making up details
                - If asked about topics not related to Trash Taste, politely redirect to podcast-related content

                Your goal is to help users access information from the podcast naturally, as if they were having a conversation with someone who has comprehensive knowledge of all Trash Taste episodes.
                """
                # - Keep responses concise and focused on the specific question asked
            )
        
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
                        "text": f"Here is the relevant context information:\n\n{combined_context}\n\nBased on this context, please answer: {query}"
                    }
                ]
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
            # Extract only the new tokens
            generation = generation[0][input_len:]
        
        # Decode the generated tokens
        response = self.processor.decode(generation, skip_special_tokens=True)
        
        return response