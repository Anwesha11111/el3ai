from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
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
        "model": "gemini-1.5-flash"
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Processing: {request.message}")
        
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing GOOGLE_AI_API_KEY")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        response = model.generate_content(request.message)
        
        return {
            "response": response.text,
            "sources": [
                "https://rbi.org.in",
                "https://sebi.gov.in",
                "https://www.incometax.gov.in"
            ]
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
