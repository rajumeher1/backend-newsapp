# app/config.py
import os
from dotenv import load_dotenv


load_dotenv()  # Load variables from .env


# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "news_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "articles")


# HTTP headers for requests
# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
#                   "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
#     "Accept-Language": "en-US,en;q=0.9",
#     "Accept-Encoding": "gzip, deflate, br",
#     "Connection": "keep-alive"
# }

# HEADERS = {
#     "User-Agent": "Mozilla/5.0"
# }

# Hugging Face API token
# HF_TOKEN = os.getenv("HF_TOKEN")
