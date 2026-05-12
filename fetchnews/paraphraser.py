# fetchnews/paraphraser.py

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
            do_sample=False,
            forced_bos_token_id=0
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )