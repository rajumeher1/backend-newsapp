# cron/main.py

import time
import numpy as np

from fetchnews.config import RSS_FEEDS
from fetchnews.rss_fetcher import get_feed_entries
from fetchnews.article_creator import process_item
from fetchnews.db import get_existing_articles, save_articles


def run():

    # Fetch existing articles and embeddings from MongoDB
    existing_articles, existing_embeddings, seen_links = get_existing_articles()

    existing_embeddings = [np.array(e) for e in existing_embeddings]

    new_articles = []

    for source, url in RSS_FEEDS.items():

        try:
            entries = get_feed_entries(url)
        except Exception as e:
            print(f"Error fetching {source}: {e}")
            continue

        for item in entries:

            try:
                result = process_item(item, source, seen_links, existing_embeddings)

                if not result:
                    print(f"Skipped item from {source}")
                    continue

                article, embedding = result

                new_articles.append(article)

                seen_links.add(article["link"])
                existing_embeddings.append(np.array(embedding))

                time.sleep(1)  # ✅ real rate limiting

            except Exception as e:
                print(f"Error processing item from {source}: {e}")

    # Save new articles to MongoDB
    save_articles(new_articles)

    print(f"New articles added: {len(new_articles)}")
    print(f"Total articles in DB: {len(existing_articles) + len(new_articles)}")


if __name__ == "__main__":
    run()