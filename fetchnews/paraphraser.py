# fetchnews/paraphraser.py

import torch
from fetchnews.models import Paraphraser


def paraphraser(text: str) -> str:

    if not text:
        return "No text"

    input_text = f"paraphrase this in a neutral tone: {text}"

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

# fetchnews/paraphraser.py

# import torch
# from fetchnews.models import Paraphraser


# def paraphraser(text: str) -> str:

#     if not text:
#         return "No text"

#     # NOTE: humarin/chatgpt_paraphraser_on_T5_base was fine-tuned specifically
#     # on prompts in the form "paraphrase: {text}". Sending a different
#     # instruction (e.g. "paraphrase this news headline in a neutral tone: ...")
#     # is out-of-distribution for the model and gives unpredictable results,
#     # since it never saw that phrasing during training.
#     input_text = f"paraphrase: {text}"

#     # load singleton model
#     tokenizer, model, device = Paraphraser.get()

#     inputs = tokenizer(
#         input_text,
#         return_tensors="pt",
#         truncation=True,
#         padding="longest"
#     ).to(device)

#     with torch.no_grad():

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=60,
#             num_beams=3,
#             do_sample=False
#             # NOTE: `forced_bos_token_id=0` was removed. That parameter forces
#             # a specific token to be generated first, which makes sense for
#             # BART/mBART-style models (where it's used to force a language
#             # token). This is a T5 model: token id 0 in T5's vocabulary is
#             # `<pad>`, not a beginning-of-sequence marker, so forcing it as
#             # the first generated token was corrupting output rather than
#             # steering it.
#         )

#     return tokenizer.decode(
#         outputs[0],
#         skip_special_tokens=True
#     )