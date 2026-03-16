# cron/main.py

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetchnews.config import RSS_FEEDS
from fetchnews.rss_fetcher import get_feed_entries
from fetchnews.article_creator import process_item
from fetchnews.db import get_existing_articles, save_articles


def run():

    # Fetch existing articles and embeddings from MongoDB
    existing_articles, existing_embeddings, seen_links = get_existing_articles()

    existing_embeddings = [np.array(e) for e in existing_embeddings]

    new_articles = []

    futures = []

    with ThreadPoolExecutor(max_workers=5) as executor:

        # Iterate through RSS feeds
        for source, url in RSS_FEEDS.items():

            entries = get_feed_entries(url)

            for item in entries:

                futures.append(
                    executor.submit(
                        process_item,
                        item,
                        source,
                        seen_links,
                        existing_embeddings
                    )
                )

        for future in as_completed(futures):

            result = future.result()

            if not result:
                continue

            article, embedding = result

            new_articles.append(article)

            seen_links.add(article["link"])
            existing_embeddings.append(np.array(embedding))

            time.sleep(5)  # optional rate limiting

    # Save new articles to MongoDB
    save_articles(new_articles)

    print(f"New articles added: {len(new_articles)}")
    print(f"Total articles in DB: {len(existing_articles) + len(new_articles)}")


if __name__ == "__main__":
    run()