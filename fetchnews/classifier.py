# fetchnews/classifier.py
import time
from fetchnews.config import CLIENT

model = "facebook/bart-large-mnli"

def classifier(text):
    text = f"News article: {text}"

    result = CLIENT.zero_shot_classification(
        text,
        labels,
        multi_label=True,
        model=model
    )

    # Use dict-style access
    # categories = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]
    categories = [r.label for r in result if r.score > 0.5]


    if not categories:
        categories = ['other']

    return categories

