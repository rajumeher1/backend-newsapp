# app/rss_fetcher.py
import feedparser
from app.config import HEADERS
import requests

def get_feed_entries(rss_url, limit=5):
    """
    Fetch RSS feed entries from a given URL.
    
    Args:
        rss_url (str): RSS feed URL
        limit (int): Maximum number of entries to return
    Returns:
        list of feedparser entries
    """
    try:
        response = requests.get(rss_url, headers=HEADERS, timeout=20)
        feed = feedparser.parse(response.content)

        # Sort by published date (newest first)
        entries_sorted = sorted(
            feed.entries,
            key=lambda e: getattr(e, "published_parsed", None),
            reverse=True
        )
        return entries_sorted[:limit]

    except Exception as e:
        print(f"Error fetching RSS feed {rss_url}: {e}")
        return []

def get_image_url(item):
    """
    Extract image URL from a feed item using common RSS fields.
    """
    if "enclosures" in item and item.enclosures:
        return item.enclosures[0].get("href")

    if "media_content" in item and item.media_content:
        return item.media_content[0].get("url")

    if "media_thumbnail" in item and item.media_thumbnail:
        return item.media_thumbnail[0].get("url")

    if "fullimage" in item and item.fullimage:
        return item.fullimage

    return None