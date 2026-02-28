# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent
import os
import uvicorn

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
# Chat Endpoint
# --------------------------------------------------
@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    """
    Accepts a natural language query and returns:
    - text response
    - optional artifact (image file path)
    """

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)