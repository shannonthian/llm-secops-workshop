import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# We keep the import so the assignment still shows HuggingFace usage
from transformers import pipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLMSecOps Workshop - QA Service",
    description="A simple Question Answering API using HuggingFace and FastAPI",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str
    context: str


class ChatResponse(BaseModel):
    answer: str


# Try to load the HuggingFace model.
# If it fails (because of SSL / corporate proxy), we fall back to a dummy mode.
qa_pipeline = None
MODEL_AVAILABLE = False

try:
    logger.info("Loading HuggingFace QA pipeline...")
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad"
    )
    MODEL_AVAILABLE = True
    logger.info("HuggingFace QA model loaded successfully.")
except Exception as e:
    # IMPORTANT: do NOT raise, just log and continue.
    logger.error("Failed to load HuggingFace model: %s", e)
    MODEL_AVAILABLE = False


@app.get("/")
async def root():
    if MODEL_AVAILABLE:
        status = "HuggingFace QA model is loaded."
    else:
        status = (
            "HuggingFace QA model is NOT available "
            "(likely due to SSL/proxy blocking access to huggingface.co)."
        )

    return {
        "message": "LLMSecOps QA API is running.",
        "model_status": status,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    If the HuggingFace model is available, use it.
    Otherwise, return a fallback message so the API still works.
    """
    try:
        if MODEL_AVAILABLE and qa_pipeline is not None:
            result = qa_pipeline(
                question=req.question,
                context=req.context,
            )
            answer = result.get("answer", "")
        else:
            # Fallback behaviour when model can't be downloaded
            answer = (
                "Model is not available in this environment "
                "(SSL/proxy issue with huggingface.co). "
                f"Echoing your question instead: '{req.question}'"
            )

        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
