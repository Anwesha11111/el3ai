from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel

app = FastAPI(title="FinLit AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health():
    return {"status": "healthy", "gemini": bool(os.getenv("GOOGLE_AI_API_KEY"))}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(request.message)
        
        return {
            "response": response.text,
            "sources": ["https://rbi.org.in", "https://sebi.gov.in"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
