# app/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db import collection

app = FastAPI(title="News Fetcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://newsapp-khaki-one.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "News Fetcher API is running"}

@app.get("/articles")
def fetch_news():
    """
    Trigger RSS fetching, paraphrasing, summarization, embedding, 
    duplicate check, and save new articles to MongoDB.
    """
    try:
        articles = list(collection.find().sort("createdAt", -1).limit(130))
        for article in articles:
            article["_id"] = str(article["_id"])
        return articles
        # return {"status": "success", "message": "News fetched and saved successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}