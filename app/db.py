# app/db.py
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def get_existing_articles():
    """
    Fetch existing articles from MongoDB.
    Returns:
        articles (list of dicts)
        embeddings (list of lists)
        links (set of URLs)
    """
    articles = list(collection.find({}))
    embeddings = [a["embedding"] for a in articles if "embedding" in a]
    links = {a["link"] for a in articles}
    return articles, embeddings, links

def save_articles(articles):
    """
    Save new articles to MongoDB.
    Args:
        articles (list of dicts)
    """
    if not articles:
        print("No new articles to save.")
        return

    collection.insert_many(articles)
    print(f"Inserted {len(articles)} new articles")