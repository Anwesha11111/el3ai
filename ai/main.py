from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import traceback

app = FastAPI(title="FinLit AI")

# CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.vercel.app", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gemini_key": bool(os.getenv("GOOGLE_AI_API_KEY")),
        "timestamp": os.getenv("TIMESTAMP", "ok")
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Received: {request.message}")
        
        # Check API key
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing GOOGLE_AI_API_KEY")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1000
            }
        )
        
        response = model.generate_content(request.message)
        
        return {
            "response": response.text,
            "sources
