# app/summarizer.py
import requests
import time
import trafilatura
from fetchnews.config import HEADERS, CLIENT


def summarizer(url: str):
    """
    Summarize the content of a URL.
    
    Args:
        url (str)
    Returns:
        str: summary text
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        html = response.text
        content = trafilatura.extract(html)

        if not content:
            return "No content"

        content = content[:4000]  # Limit to first 4000 characters

        last_error = None

        # Retry mechanism
        for _ in range(3):
            try:
                result = CLIENT.summarization(
                    content,
                    # model="facebook/bart-large-cnn"
                    model="sshleifer/distilbart-cnn-12-6"
                )
                return result.summary_text.replace("<n>", " ").strip()

            except Exception as e:
                last_error = e
                time.sleep(5)

        return "Summary unavailable"

    except Exception as e:
        print("Error fetching/summarizing URL:", e)
        return "Summary unavailable"