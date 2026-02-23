from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.intents import detect_intent
from app.guardrails import redact_pii, safety_preamble, account_boundary
from app.rag import load_chunks, load_index, retrieve, build_prompts, generate_with_groq, format_citations

app = FastAPI(title="Travelers RAG Demo Bot (Groq + FAISS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/")
def home():
    return FileResponse("web/index.html")

@app.get("/chat")
def chat_info():
    return {"hint": "Use POST /chat with JSON {message, user_type}"}

class ChatRequest(BaseModel):
    message: str
    user_type: str = "personal"

class ChatResponse(BaseModel):
    intent: str
    answer: str
    citations: list[dict]
    pii_redacted: bool

chunks = None
index = None

@app.on_event("startup")
def startup():
    global chunks, index
    chunks = load_chunks(settings.CHUNKS_PATH)
    index = load_index(settings.FAISS_PATH)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    cleaned, changed = redact_pii(req.message)
    intent = detect_intent(cleaned)
    sources = retrieve(index, chunks, cleaned, top_k=settings.TOP_K)
    sys_prompt, user_prompt = build_prompts(cleaned, sources, req.user_type, intent)
    answer = generate_with_groq(sys_prompt, user_prompt)
    answer = safety_preamble() + account_boundary() + answer
    citations = format_citations(sources, max_cites=3)
    return ChatResponse(
        intent=intent,
        answer=answer,
        citations=citations,
        pii_redacted=changed
    )
