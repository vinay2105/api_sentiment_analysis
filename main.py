from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class Review(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/analyze/")
def analyze_sentiment(review: Review):
    if review.text.strip():
        result = sentiment_pipeline(review.text)[0]
        return {
            "sentiment": result["label"],
            "confidence": round(result["score"], 2),
        }
    else:
        return {"error": "Input text cannot be empty."}
