import uuid
import os
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import lancedb
import time

from models.gemma import Gemma3
from models.gemini import Gemini
from rag.rag_pipeline import rag_pipeline

app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware to allow cross-origin requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
db = lancedb.connect(os.path.join(parent_dir, "srt_embeddings","transcripts_lancedb"))
table = db.open_table("transcripts")
# embed_model_path = "Alibaba-NLP/gte-modernbert-base"
embed_model_path = "/Users/dave/AI/models/gte-modernbert-base"
modern_colbert_path = "/Users/dave/AI/models/GTE-ModernColBERT-v1"


gemini_api_key = os.environ.get("GEMINI_API_KEY")
gemini_model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-04-17")
llm_path = "/Users/dave/AI/models/gemma-3-4b-it"

if gemini_api_key:
    try:
        generator = Gemini(api_key=gemini_api_key, model_name=gemini_model_name)
        print(f"Successfully initialized Gemini using API with model: {gemini_model_name}")
    except Exception as e:
        print(f"Failed to initialize Gemini: {str(e)}. Falling back to Gemma3.")
        generator = Gemma3(model_path=llm_path)
else:
    print("No Gemini API key found. Using local Gemma3 model.")
    generator = Gemma3(model_path=llm_path)

chat_sessions = {}

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class Source(BaseModel):
    title: str
    url: str
    text: str

class ChatCompletionRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    temperature: float = 0.3
    max_new_tokens: int = 1000
    use_query_rewriting: bool = True

class ChatCompletionResponse(BaseModel):
    session_id: str
    message: Message
    sources: List[Source]
    used_rewritten_query: Optional[bool] = False
    rewritten_query: Optional[str] = None

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Message] = []
    
    def add_message(self, role: str, content: str):
        self.history.append(Message(role=role, content=content))
    
    def get_history_as_string(self):
        """Convert chat history to a string format for context"""
        history_text = ""
        for msg in self.history:
            history_text += f"{msg.role.upper()}: {msg.content}\n"
        return history_text

@app.post("/chat", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = ChatSession(session_id)
    elif session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession(session_id)
    
    session = chat_sessions[session_id]
    
    session.add_message("user", request.message)
    
    history_context = session.get_history_as_string()
    
    used_rewritten_query = False
    rewritten_query = None
    query = request.message
    
    if request.use_query_rewriting and len(session.history) > 1:
        rewritten_query = generator.rewrite_query(
            current_query=request.message,
            conversation_history=history_context,
            temperature=0.0
        )
        print(rewritten_query)
        if rewritten_query != request.message:
            query = rewritten_query
            used_rewritten_query = True
            print(f"Original query: '{request.message}'")
            print(f"Rewritten query: '{rewritten_query}'")
        else:
            rewritten_query = None

    try:
        result = rag_pipeline(
            query=query,
            conversation_history=history_context, 
            generator=generator,
            # embedding_model_path=embed_model_path,
            modern_colbert_path=modern_colbert_path,
            retrieve_type="colbert",
            table=table,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            do_sample=True
        )

        session.add_message("assistant", result['response'])
        
        sources = []
        for file_name, source_data in result['sources'].items():
            for idx in range(len(source_data['url'])):
                sources.append(Source(
                    title=file_name,
                    url=source_data['url'][idx],
                    text=source_data['text'][idx]
                ))
        
        return ChatCompletionResponse(
            session_id=session_id,
            message=Message(role="assistant", content=result['response']),
            sources=sources,
            used_rewritten_query=used_rewritten_query,
            rewritten_query=rewritten_query if used_rewritten_query else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "history": chat_sessions[session_id].history}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "success", "message": "Session deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)