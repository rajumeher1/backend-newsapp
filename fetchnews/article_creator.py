#fetchnes/article_creator.py

from datetime import datetime, timezone
from fetchnews.rss_fetcher import get_image_url
from fetchnews.paraphraser import paraphraser
from fetchnews.summarizer import summarizer
from fetchnews.classifier import classifier
from fetchnews.embeddings import create_embedding, is_duplicate, create_title_embedding


def process_item(item, source, seen_links, existing_embeddings):
    link = item.get("link")
    title = item.get("title")

    if not title or not link:
        return None

    if link in seen_links:
        return None

    try:
        title_embedding = create_title_embedding(title)

        if is_duplicate(title_embedding, existing_embeddings):
            return None

        image_url = get_image_url(item)

        new_title = paraphraser(title) or title

        if not new_title or new_title in ["No title", "Title unavailable"]:
            return None

        summary = summarizer(link)

        if not summary or summary in ["No content", "Summary unavailable"]:
            return None

        text = new_title + ". " + summary

        categories = classifier(text)

        embedding = create_embedding(text)

        if is_duplicate(embedding, existing_embeddings):
            print(f"Duplicate article skipped: {title}")
            return None

        article = {
            "image": image_url,
            "title": new_title,
            "summary": summary,
            "link": link,
            "publishedAt": item.get("published", ""),
            "source": source,
            "category": categories,
            "embedding": embedding.tolist(),
            "createdAt": datetime.now(timezone.utc)
        }

        return article, embedding

    except Exception as e:
        print("Processing Failed:", e)
        return None
