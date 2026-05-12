# # app/paraphraser.py

# import time
# from fetchnews.config import CLIENT

# model = "meta-llama/Meta-Llama-3-8B-Instruct"

# def paraphraser(title: str) -> str:
#     """
#     Paraphrase a news title using HF model
#     """

#     if not title:
#             return "No title"

#     messages = [
#       {"role": "system",
#       "content": '''You are a professional paraphrasing assistant.
#                     Rewrite the user's text keeping the same meaning. Output only the
#                     paraphrased text, below or within 20 words, without a period at the end.'''},
#       {"role": "user", "content": title}
#     ]

#     last_error = None

#     for _ in range(2):

#         try:
#             response = CLIENT.chat.completions.create(
#                 model = model,
#                 messages = messages,
#                 temperature = 0.4
#             )

#             rewritten = response.choices[0].message.content
#             return rewritten.strip()

#         except Exception as e:
#             last_error = e
#             time.sleep(5)

#     print(f'Error for title: {title} - {last_error}')
            
#     return "Title unavailable"

# app/paraphraser.py

import torch

from fetchnews.models import Paraphraser


def paraphraser(text: str) -> str:

    if not text:
        return "No text"

    input_text = f"paraphrase this news headline in a neutral tone: {text}"

    # load singleton model
    tokenizer, model, device = Paraphraser.get()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="longest"
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=3,
            do_sample=False
            forced_bos_token_id=0
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )