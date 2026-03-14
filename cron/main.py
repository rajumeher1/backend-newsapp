# cron/main.py
import time
from app.config import RSS_FEEDS
from fetchnews.rss_fetcher import get_feed_entries, get_image_url
from fetchnews.paraphraser import paraphraser
from fetchnews.summarizer import summarizer
from fetchnews.embeddings import create_embedding, is_duplicate
from fetchnews.db import get_existing_articles, save_articles
import numpy as np
from datetime import datetime, timezone

def run():
    # Fetch existing articles and embeddings from MongoDB
    existing_articles, existing_embeddings, seen_links = get_existing_articles()
    # Convert embeddings to numpy arrays for similarity check
    existing_embeddings = [np.array(e) for e in existing_embeddings]

    new_articles = []

    # Iterate through RSS feeds
    for source, url in RSS_FEEDS.items():
        entries = get_feed_entries(url, limit=3)

        for item in entries:
            link = item.get("link")
            title = item.get("title")

            # Skip already seen links
            if link in seen_links:
                continue

            # Process article
            image_url = get_image_url(item)
            new_title = paraphraser(title)
            summary = summarizer(link)
            embedding = create_embedding(new_title, summary)

            # Check for semantic duplicates
            if is_duplicate(embedding, existing_embeddings):
                print(f"Duplicate article skipped: {title}")
                continue

            article = {
                "image": image_url,
                "title": new_title,
                "description": summary,
                "link": link,
                "publishedAt": item.get("published", ""),
                "source": source,
                "category": [],
                "embedding": embedding.tolist(),
                "createdAt": datetime.now(timezone.utc)
            }

            new_articles.append(article)
            seen_links.add(link)
            existing_embeddings.append(embedding)  # Update memory for duplicates

            # Optional rate limiting
            time.sleep(5)

    # Save new articles to MongoDB
    save_articles(new_articles)
    print(f"New articles added: {len(new_articles)}")
    print(f"Total articles in DB: {len(existing_articles) + len(new_articles)}")


if __name__ == "__main__":
    run()