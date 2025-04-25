from typing import List, Dict, Any, Optional
import requests
import os
import json

class Gemini:
    _instance = None
    
    def __new__(cls, api_key=None, model_name=None):
        if cls._instance is None:
            cls._instance = super(Gemini, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash-preview-04-17"
    ):
        """
        Initialize the Gemini generator with API access.
        
        Args:
            api_key: API key for accessing Gemini API. If None, will look for GEMINI_API_KEY in environment
            model_name: Gemini model identifier to use (e.g., "gemini-2.5-flash-preview-04-17", "gemini-1.5-flash")
        """
        if not self.initialized:
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("No API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            
            self.model_name = model_name
            self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
            
            print(f"Initialized Gemini API client for model: {model_name}")
            
            self.initialized = True
    
    def _call_api(self, messages, temperature=0.0, max_output_tokens=100, top_p=0.95, top_k=40, thinking_budget: int = 0):
        """
        Make an API call to the Gemini API.
        
        Args:
            messages: List of message dictionaries to send to the API
            temperature: Sampling temperature (0.0 to 1.0)
            max_output_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            thinking_budget: Budget for thinking time (Defaults to 0; added under generationConfig.thinkingConfig)
            
        Returns:
            API response text or error message
        """
        # Build the URL without including the API key in any logging/error messages
        url = f"{self.api_base_url}/{self.model_name}:generateContent"
        
        # Parameters for the request
        params = {"key": self.api_key}
        
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": top_p,
            "topK": top_k,
            "thinkingConfig": {
                "thinkingBudget": thinking_budget
            }
        }

        payload = {
            "contents": messages,
            "generationConfig": generation_config
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, params=params, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            return part["text"]
            
            return "Error: Could not extract response text"
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
            return f"API Error: Request failed with status code {status_code}. Please check your API configuration."
        except requests.exceptions.ConnectionError:
            return "API Error: Connection failed. Please check your internet connection and try again."
        except requests.exceptions.Timeout:
            return "API Error: Request timed out. Please try again later."
        except requests.exceptions.RequestException:
            return "API Error: Request failed. Please check your API configuration."
        except json.JSONDecodeError:
            return "API Error: Invalid response format received from the API."
        except Exception as e:
            return f"Error: An unexpected error occurred: {type(e).__name__}"
        
    def generate(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        conversation_history: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 100,
        system_prompt: Optional[str] = None,
        do_sample: bool = False,
        thinking_budget: int = 0
    ) -> str:
        """
        Generate a response using Gemini based on the query, conversation history and retrieved contexts.
        
        Args:
            query: User query string
            retrieved_contexts: List of context dictionaries retrieved from the database
            conversation_history: Optional string containing conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the model behavior
            do_sample: Whether to use sampling for generation
            thinking_budget: Budget for thinking time. Defaults to 0.
            
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
        print(conversation_history)
        if conversation_history:
            user_message = f"Conversation history:\n{conversation_history}\n\nAnswer the current question about Trash Taste: {query}"

        messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"System: {system_prompt}\n\nUse this information (but never reference it):\n\n{combined_context}\n\n{user_message}\n\nSystem: {system_prompt}"
                    }
                ]
            }
        ]
        
        response = self._call_api(
            messages=messages,
            temperature=temperature if do_sample else 0.0,
            max_output_tokens=max_new_tokens,
            top_p=0.95 if do_sample else 1.0,
            top_k=40 if do_sample else 0,
            thinking_budget=thinking_budget
        )
        
        return response
    
    def rewrite_query(
        self,
        current_query: str,
        conversation_history: str,
        temperature: float = 0.0,
        max_new_tokens: int = 1000,
        do_sample: bool = False,
        thinking_budget: int = 0
    ) -> str:
        """
        Rewrite an ambiguous or context-dependent query into a self-contained query
        using the Gemini model, based on conversation history string.
        
        Args:
            current_query: The latest user query that may contain contextual references
            conversation_history: String containing the conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate for the rewritten query
            do_sample: Whether to use sampling for generation
            thinking_budget: Budget for thinking time. Defaults to 0.
            
        Returns:
            A rewritten query that is self-contained, or the original query if not context-dependent
        """
        system_prompt = """
        You are an AI assistant whose ONLY task is to decide whether the user's *current* query is context-dependent and, if so, rewrite it into a fully self-contained version. **Err on the side of rewriting**—when in doubt, expand the query.
        
        **Rewriting rules**
        1. If the current query depends on previous messages (e.g. pronouns like "he", "it", "them"; adverbs like "here", "there"; elliptical phrases like "tell me more", "what about the second one?"), rewrite the query so that every reference becomes explicit. Keep the original wording and tone as much as possible.
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
                "role": "user",
                "parts": [
                    {
                        "text": f"{system_prompt}\n\n{user_message}"
                    }
                ]
            }
        ]
        
        rewritten_query = self._call_api(
            messages=messages,
            temperature=temperature if do_sample else 0.0,
            max_output_tokens=max_new_tokens,
            top_p=0.95 if do_sample else 1.0,
            top_k=40 if do_sample else 0,
            thinking_budget=thinking_budget
        )
        
        rewritten_query = rewritten_query.strip()
        
        return rewritten_query