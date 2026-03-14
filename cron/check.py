import requests
import feedparser

# rss1 = "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"

rss1 = "https://www.hindustantimes.com/feeds/rss/latest/rssfeed.xml"
# rss1 = "https://indianexpress.com/section/news-today/feed/"
# rss1 = "https://publish.tribuneindia.com/newscategory/top-headlines/feed/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
#                   "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
#     "Accept-Language": "en-US,en;q=0.9",
#     "Accept-Encoding": "gzip, deflate, br",
#     "Connection": "keep-alive"
# }

response = requests.get(rss1, headers=HEADERS)

# print(response.status_code)
# print(response.headers.get("content-type"))
# print(response.text[:500])

feed = feedparser.parse(response.content)

print("Entries:", len(feed.entries))

sorted_entries = sorted(feed.entries, key= lambda e: getattr(e, 'published_parsed', None), reverse=True)

print(sorted_entries[:1])