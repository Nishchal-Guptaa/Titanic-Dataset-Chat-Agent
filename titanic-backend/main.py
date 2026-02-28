# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent
import os
import uvicorn
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Chat Agent API")

# --------------------------------------------------
# CORS (important for Streamlit frontend)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your Streamlit URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------
class QueryRequest(BaseModel):
    query: str


# --------------------------------------------------
# Health Check Route
# --------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "Titanic Chat Agent API is running 🚢"}


# --------------------------------------------------
# Diagnostic Endpoint - tells you what's missing
# --------------------------------------------------
@app.get("/debug")
def debug_check():
    checks = {}
    checks["GROQ_API_KEY"] = "✅ set" if os.getenv("GROQ_API_KEY") else "❌ missing"
    checks["HF_TOKEN"] = "✅ set" if os.getenv("HF_TOKEN") else "❌ missing"
    checks["LANGSMITH_API_KEY"] = "✅ set" if os.getenv("LANGSMITH_API_KEY") else "⚠️ missing (optional)"
    checks["PORT"] = os.getenv("PORT", "not set (default 8000)")

    try:
        import pandas as pd
        checks["pandas"] = "✅ OK"
    except Exception as e:
        checks["pandas"] = f"❌ {e}"

    try:
        from langchain_groq import ChatGroq
        checks["langchain_groq"] = "✅ OK"
    except Exception as e:
        checks["langchain_groq"] = f"❌ {e}"

    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        checks["langchain_huggingface"] = "✅ OK"
    except Exception as e:
        checks["langchain_huggingface"] = f"❌ {e}"

    try:
        from langchain_chroma import Chroma
        checks["langchain_chroma"] = "✅ OK"
    except Exception as e:
        checks["langchain_chroma"] = f"❌ {e}"

    try:
        from langchain.agents import create_agent
        checks["langchain_create_agent"] = "✅ OK"
    except Exception as e:
        checks["langchain_create_agent"] = f"❌ {e}"

    return {"diagnostics": checks}


# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------
@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    """
    Accepts a natural language query and returns:
    - text response
    - optional artifact (image file path)
    """
    try:
        result = run_agent(request.query)

        # Extract final AI message
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None

        text_response = ""
        artifact = None

        if final_message:
            text_response = final_message.content

            # If tool returned artifact, it will be inside additional_kwargs
            if hasattr(final_message, "additional_kwargs"):
                artifact = final_message.additional_kwargs.get("artifact")

        return {
            "text": text_response,
            "artifact": artifact
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return {
            "text": f"Error: {str(e)}",
            "artifact": None,
            "error": traceback.format_exc()
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)