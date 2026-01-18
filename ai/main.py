import google.generativeai as genai
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FinLit AI")

# CORS for all origins (Vercel + browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health():
    """Health check endpoint"""
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    return {
        "status": "healthy",
        "api_key_set": api_key is not None,
        "timestamp": "2026-01-18"
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Validate API key
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_AI_API_KEY not set")
        
        print(f"Processing message: {request.message[:50]}...")
        
        # Configure Gemini (2026 stable models)
        genai.configure(api_key=api_key)
        
        # TRY multiple models (fallback chain)
        models_to_try = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash-latest", 
            "gemini-pro"
        ]
        
        response = None
        for model_name in models_to_try:
            try:
                print(f"Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(request.message)
                print(f"Success with {model_name}")
                break
            except Exception as model_error:
                print(f"Model {model_name} failed: {model_error}")
                continue
        
        if not response:
            raise HTTPException(status_code=500, detail="All Gemini models failed")
        
        return {
            "response": response.text,
            "sources": [
                "https://rbi.org.in",
                "https://sebi.gov.in", 
                "https://incometaxindia.gov.in"
            ]
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
