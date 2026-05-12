# fetchnews/article_creator.py

import time
from datetime import datetime, timezone

from fetchnews.rss_fetcher import get_image_url
from fetchnews.paraphraser import paraphraser
from fetchnews.summarizer import summarizer
from fetchnews.classifier import classifier
from fetchnews.embeddings import (
    create_embedding,
    is_duplicate,
    create_title_embedding
)


def process_item(item, source, seen_links, existing_embeddings):

    link = item.get("link")
    title = item.get("title")

    # -----------------------------
    # 1️⃣ FAST FILTERS FIRST (cheap checks)
    # -----------------------------
    if not title or not link:
        return None

    if link in seen_links:
        return None

    # -----------------------------
    # 2️⃣ LIGHT EMBEDDING FIRST (cheap model)
    # -----------------------------
    title_embedding = create_title_embedding(title)

    if is_duplicate(title_embedding, existing_embeddings):
        return None

    try:
        # -----------------------------
        # 3️⃣ IMAGE (external call, cheap-ish)
        # -----------------------------
        image_url = get_image_url(item)

        # -----------------------------
        # 4️⃣ HEAVY NLP PIPELINE STARTS HERE
        # -----------------------------

        new_title = paraphraser(title)

        if not new_title or new_title in ["No title", "Title unavailable"]:
            return None

        summary = summarizer(link)

        if not summary or summary in ["No content", "Summary unavailable"]:
            return None

        text = f"{new_title}. {summary}"

        # -----------------------------
        # 5️⃣ CLASSIFICATION (medium-heavy)
        # -----------------------------
        categories = classifier(text)

        # -----------------------------
        # 6️⃣ FINAL EMBEDDING (used for DB + dedup)
        # -----------------------------
        embedding = create_embedding(text)

        if is_duplicate(embedding, existing_embeddings):
            return None

        # -----------------------------
        # 7️⃣ BUILD ARTICLE
        # -----------------------------
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