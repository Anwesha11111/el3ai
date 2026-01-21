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

SYSTEM_PROMPT = """
You are FinLit Bot, a financial literacy assistant for beginners in India.

Rules:
- Keep answers short (4–5 lines max) and give gap between lines
- Use simple conversational language
- Do NOT use markdown, tables, or headings
- Focus on India (RBI, SEBI, Income Tax, UIDAI)
- Do NOT give legal or investment advice, only education

When helpful:
- Suggest up to 2 relevant official websites (RBI, SEBI, Income Tax, UIDAI, NSDL, Zerodha Varsity)
- Suggest 1 clickable YouTube search link (not a specific video)
- Clearly label them as "Helpful links"

End every reply with ONE short follow-up question.
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_set": bool(os.getenv("GOOGLE_AI_API_KEY")),
        "model": "gemini-3-flash-preview"
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing API key")

        client = genai.Client(api_key=api_key)

        final_prompt = f"""{SYSTEM_PROMPT}

User question:
{request.message}
"""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": final_prompt}]
                }
            ]
        )

        # SAFELY extract text
        ai_text = response.candidates[0].content.parts[0].text

        return {
            "response": ai_text,
            "sources": [
                "https://rbi.org.in",
                "https://sebi.gov.in",
                "https://incometaxindia.gov.in"
            ]
        }

    except Exception as e:
        print("AI ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
