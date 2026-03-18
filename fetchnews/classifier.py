# fetchnews/classifier.py

from fetchnews.config import CLIENT
import requests

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
    print(type(result))
    print(result)

    # categories = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]

    categories = [r.label for r in result if r.score > 0.5]
    # categories = [r["label"] for r in resultif r["score"] > 0.5]

    if not categories:
        categories = ['other']

    # print(categories)

    return categories

# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# headers = {
#     "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
# }

# def classifier(text):
   
#     payload = {
#         "inputs": f"News article: {text}",
#         "parameters": {"candidate_labels": labels, "multi_label": True}
#     }

#     output = requests.post(API_URL, headers=headers, json=payload, timeout=2)

#     result = output.json()

#     # Extract labels with score > threshold
#     # Extract labels with score > 0.5
#     if "labels" in result and "scores" in result:
#         categories = [
#             label for label, score in zip(result["labels"], result["scores"])
#             if score > 0.4
#         ]
#     else:
#         categories = []

#     # Fallback if no labels meet threshold
#     if not categories:
#         categories = ['other']

#     return categories

# Example usage
# text = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
# print(classifier(text))