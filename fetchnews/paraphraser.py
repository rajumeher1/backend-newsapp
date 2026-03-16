# app/paraphraser.py

import time
from fetchnews.config import CLIENT

model = "meta-llama/Meta-Llama-3-8B-Instruct"

def paraphraser(title: str) -> str:
    """
    Paraphrase a news title using HF model
    """

    if not title:
            return "No title"

    messages = [
      {"role": "system",
      "content": '''You are a professional paraphrasing assistant.
                    Rewrite the user's text keeping the same meaning. Output only the
                    paraphrased text. within or below 20 words, without a period at the end.'''},
      {"role": "user", "content": title}
    ]

    last_error = None

    for _ in range(2):

        try:
            response = CLIENT.chat.completions.create(
                model = model,
                messages = messages,
                temperature = 0.4
            )

            rewritten = response.choices[0].message.content
            return rewritten.strip()

        except Exception as e:
            last_error = e
            time.sleep(5)

    print(f'Error for title: {title} - {last_error}')
            
    return "Title unavailable"