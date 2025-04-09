import uuid
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import lancedb
import time

from models.gte_modernbert import GteModernbert
from models.gemma import Gemma3
from rag.retrieval import retrieve_context
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
db = lancedb.connect("./srt_embeddings/transcripts_lancedb")
table = db.open_table("transcripts")
embed_model_path = "/Users/dave/AI/models/gte-modernbert-base"
embedding_model = GteModernbert(embed_model_path)
llm_path = "/Users/dave/AI/models/gemma-3-4b-it"
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
    max_new_tokens: int = 500
    top_k: int = 20

class ChatCompletionResponse(BaseModel):
    session_id: str
    message: Message
    sources: List[Source]

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
    augmented_query = f"Chat history:\n{history_context}\nCurrent question: {request.message}"
    
    try:
        result = rag_pipeline(
            query=augmented_query,
            retriever_function=retrieve_context,
            generator=generator,
            embedding_model=embedding_model,
            table=table,
            top_k=request.top_k,
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
            sources=sources
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