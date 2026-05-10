# # cron/main.py

# import numpy as np

# from fetchnews.config import RSS_FEEDS
# from fetchnews.rss_fetcher import get_feed_entries
# from fetchnews.article_creator import process_item
# from fetchnews.db import get_existing_articles, save_articles


# def run():

#     # Fetch existing articles and embeddings from MongoDB
#     existing_articles, existing_embeddings, seen_links = get_existing_articles()

#     existing_embeddings = [np.array(e) for e in existing_embeddings]

#     new_articles = []

#     for source, url in RSS_FEEDS.items():

#         try:
#             entries = get_feed_entries(url)
#         except Exception as e:
#             print(f"Error fetching {source}: {e}")
#             continue

#         for item in entries:

#             try:
#                 result = process_item(item, source, seen_links, existing_embeddings)

#                 if not result:
#                     print(f"Skipped item from {source}")
#                     continue

#                 article, embedding = result

#                 new_articles.append(article)

#                 seen_links.add(article["link"])
#                 existing_embeddings.append(np.array(embedding))


#             except Exception as e:
#                 print(f"Error processing item from {source}: {e}")

#     # Save new articles to MongoDB
#     save_articles(new_articles)

#     print(f"New articles added: {len(new_articles)}")
#     print(f"Total articles in DB: {len(existing_articles) + len(new_articles)}")


# if __name__ == "__main__":
#     run()

# cron/main.py

import numpy as np
import logging

from fetchnews.config import RSS_FEEDS
from fetchnews.rss_fetcher import get_feed_entries
from fetchnews.article_creator import process_item
from fetchnews.db import get_existing_articles, save_articles


# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def run():
    logger.info("Starting RSS cron job...")

    # -----------------------------
    # Load existing data from DB
    # -----------------------------
    existing_articles, existing_embeddings, seen_links = get_existing_articles()

    # Convert embeddings to numpy once (immutable reference)
    try:
        existing_embeddings = [np.array(e) for e in existing_embeddings]
    except Exception:
        logger.exception("Failed to parse existing embeddings")
        existing_embeddings = []

    # IMPORTANT:
    # Use a runtime copy so we don't mutate DB-loaded structure
    run_embeddings = existing_embeddings.copy()

    new_articles = []

    # -----------------------------
    # Process each RSS feed
    # -----------------------------
    for source, url in RSS_FEEDS.items():
        logger.info(f"Fetching feed: {source}")

        try:
            entries = get_feed_entries(url)
        except Exception:
            logger.exception(f"Error fetching feed: {source}")
            continue

        for item in entries:

            link = item.get("link")

            # -----------------------------
            # Fast duplicate skip (before processing)
            # -----------------------------
            if not link or link in seen_links:
                continue

            try:
                result = process_item(
                    item,
                    source,
                    seen_links,
                    run_embeddings
                )

                if result is None:
                    logger.debug(
                        "Skipped item (process_item returned None) | source=%s | link=%s",
                        source,
                        item.get("link")
                    )
                    continue

                article, embedding = result

                new_articles.append(article)

                # Update runtime state only
                seen_links.add(article["link"])
                run_embeddings.append(np.array(embedding))

            except Exception:
                logger.exception(f"Error processing item from {source}")

    # -----------------------------
    # Save to DB (bulk insert)
    # -----------------------------
    try:
        if new_articles:
            save_articles(new_articles)
            logger.info(f"Saved {len(new_articles)} new articles")
        else:
            logger.info("No new articles to save")
    except Exception:
        logger.exception("Failed to save articles")

    logger.info(
        "Job finished | New: %d | Total approx: %d",
        len(new_articles),
        len(existing_articles) + len(new_articles)
    )


if __name__ == "__main__":
    run()