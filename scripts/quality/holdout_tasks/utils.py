# HellaSwag preprocessing, same as lm_eval/tasks/hellaswag/utils.py (lm-eval 0.4.9.1).
import re


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset):
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {"query": preprocess(doc["activity_label"] + ": " + ctx),
                "choices": [preprocess(e) for e in doc["endings"]], "gold": int(doc["label"])}

    return dataset.map(_process_doc)
