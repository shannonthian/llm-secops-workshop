from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="LLMSecOps Workshop - QA Service",
    description="A simple Question Answering API using HuggingFace and FastAPI",
    version="1.0.0",
)

# Initialise HuggingFace QA pipeline
# Model: distilbert-base-uncased-distilled-squad
try:
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad"
    )
except Exception as e:
    # If model download fails at startup, raise clear error
    raise RuntimeError(f"Failed to load QA model: {e}")


class ChatRequest(BaseModel):
    question: str
    context: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    return {
        "message": "LLMSecOps QA API is running. POST /chat with question & context."
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Accepts a question and context, returns the model's answer.
    """
    try:
        result = qa_pipeline(
            question=req.question,
            context=req.context
        )
        return ChatResponse(answer=result.get("answer", ""))
    except Exception as e:
        # In real production, avoid returning raw error messages
        raise HTTPException(status_code=500, detail=f"Model error: {e}")
