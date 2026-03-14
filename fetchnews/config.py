# app/config.py
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()  # Load variables from .env


# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "news_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "articles")

# RSS Feeds
RSS_FEEDS = {
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/latest/rssfeed.xml",
    "Indian Express": "https://indianexpress.com/section/news-today/feed/",
    "The Tribune": "https://publish.tribuneindia.com/newscategory/top-headlines/feed/",
}

# HTTP headers for requests
# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
#                   "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
#     "Accept-Language": "en-US,en;q=0.9",
#     "Accept-Encoding": "gzip, deflate, br",
#     "Connection": "keep-alive"
# }

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Hugging Face API token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Initialize Hugging Face inference client
CLIENT = InferenceClient(
    provider="auto",
    api_key=HUGGINGFACE_API_TOKEN
)