# fetchnews/classifier.py

from fetchnews.config import CLIENT

model = "facebook/bart-large-mnli"

labels = ["India", "World", "Business", "Sports", "Entertainment", "Finance",
        "Technology", "Politics", "Health", "Science", "Education"]

def classifier(text):
    text = f"News article: {text}"

    result = CLIENT.zero_shot_classification(
        text,
        labels,
        multi_label=True,
        model=model
    )
    
    categories = [r.label for r in result if r.score > 0.7]

    if not categories:
        return ["Misc"]

    return categories