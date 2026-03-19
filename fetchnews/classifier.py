# fetchnews/classifier.py

from fetchnews.config import CLIENT

model = "facebook/bart-large-mnli"

labels = ["india", "international", "business", "sports", "entertainment", "technology", "politics", "health", "science"]

def classifier(text):
    text = f"News article: {text}"

    result = CLIENT.zero_shot_classification(
        text,
        labels,
        multi_label=True,
        model=model
    )
    
    categories = [r.label for r in result if r.score > 0.5]

    if not categories:
        categories = ['other']

    return categories