import google.generativeai as genai
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Use the 2026 stable identifier for Flash
        model = genai.GenerativeModel("gemini-1.5-flash-latest") 
        
        response = model.generate_content(request.message)
        return {"response": response.text, "sources": ["https://rbi.org.in"]}
    except Exception as e:
        # This will show the EXACT error in your Render logs
        print(f"CRITICAL ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))
