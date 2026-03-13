# app/paraphraser.py
from config import client


model = "Qwen/Qwen2.5-7B-Instruct"


def paraphraser(title: str) -> str:
    """
    Paraphrase a news title using T5 model
    """

    messages = [
      {"role": "system", "content": "You are a professional paraphrasing assistant. Rewrite the user's text to be different but keep the same meaning. Output only the result."},
      {"role": "user", "content": title}
    ]

    try:
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = 25,
            temperature = 0.8
        )

        rewritten = response.choices[0].message.content
        return rewritten.strip().strip('"')

    except Exception as e:
        return f"Error during paraphrasing: {e}"