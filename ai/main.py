from google import genai
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FinLit AI")

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
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    return {
        "status": "healthy",
        "api_key_set": api_key is not None,
        "model": "gemini-3-flash-preview"
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing API key")
        
        client = genai.Client(api_key=api_key)

        final_prompt = f"""
You are FinLit Bot, a financial literacy assistant for beginners in India.

Rules:
- Answer briefly (4–5 lines max)
- No markdown, tables, headings, or bullet points
- Use simple language
- End with one follow-up question

User question:
{request.message}
"""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=final_prompt
        )
        
        print(f"Success! Response: {response.text[:100]}...")
        
        return {
            "response": response.text,
            "sources": [
                "https://rbi.org.in",
                "https://sebi.gov.in",
                "https://incometaxindia.gov.in"
            ]
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

