# fetchnews/summarizer.py

import requests
import trafilatura
import torch
import logging

from fetchnews.config import HEADERS
from fetchnews.models import Summarizer

logging.basicConfig(level=logging.INFO)

def summarizer(url: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)

        html = response.text

        content = trafilatura.extract(html)

        if not content:
            return "No content"

        # reduce processing size
        # content = content[:4000]

        tokenizer, model, device = Summarizer.get()

        inputs = tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="longest"
        ).to(device)

        with torch.no_grad():

            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=110,
                min_length=60,
                num_beams=1,
                do_sample=False
            )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary.strip()

    except Exception as e:
        logging.error("Error fetching/summarizing URL:", e)
        # print("Error fetching/summarizing URL:", e)
        return "Summary unavailable"