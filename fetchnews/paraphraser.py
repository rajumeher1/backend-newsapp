# app/paraphraser.py
from app.config import client


model = "Qwen/Qwen2.5-7B-Instruct"


def paraphraser(title: str) -> str:
    """
    Paraphrase a news title using T5 model
    """

    messages = [
      {"role": "system",
      "content": '''You are a professional paraphrasing assistant.
                    Rewrite the user's text in english to be different but keep the same meaning and avoid long title.
                    Output only the result without a fullstop (.) at the end.'''},
      {"role": "user", "content": title}
    ]

    try:
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = 0.8
        )

        rewritten = response.choices[0].message.content
        return rewritten.strip().strip('"')

    except Exception as e:
        return f"Error during paraphrasing: {e}"