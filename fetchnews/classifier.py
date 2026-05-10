# # fetchnews/classifier.py
# from fetchnews.config import CLIENT

# # model = "facebook/bart-large-mnli"
# model = "valhalla/distilbart-mnli-12-3"

# labels = ["India", "World", "Business", "Sports", "Entertainment", "Finance",
#         "Technology", "Politics", "Health & Fitness", "Science", "Education"]

# def classifier(text):
#     text = f"News article: {text}"

#     result = CLIENT.zero_shot_classification(
#         text,
#         labels,
#         multi_label=True,
#         model=model
#     )
    
#     categories = [r.label for r in result if r.score > 0.7]

#     if not categories:
#         return ["Misc"]

#     return categories

# fetchnews/classifier.py

from transformers import pipeline

model = "facebook/bart-large-mnli"

labels = [
    "India",
    "World",
    "Business",
    "Sports",
    "Entertainment",
    "Finance",
    "Technology",
    "Politics",
    "Health & Fitness",
    "Science",
    "Education"
]

# load once
classifier_model = pipeline(
    "zero-shot-classification",
    model=model
)

def classifier(text):

    text = f"News article: {text}"

    result = classifier_model(
        text,
        candidate_labels=labels,
        multi_label=True
    )

    categories = [
        label
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.7
    ]

    if not categories:
        return ["Misc"]

    return categories