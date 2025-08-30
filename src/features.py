import pandas as pd
from transformers import AutoTokenizer, pipeline
import re

def assign_label(row):
    if row["looks_promo_heur"]:
        return "Advertisement"
    elif row["looks_rant_no_visit_heur"]:
        return "Rant_No_Visit"
    else:
        return None


PROMPT_TEMPLATE = """Classify the following review into exactly ONE of these labels:
- Advertisement
- Irrelevant
- Rant_No_Visit
- Valid

Reply with just the label on a single line. No extra words.

Review:
{text}
Answer:"""

classifier = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

def prompt_classify(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Valid"  # safe fallback

    prompt = PROMPT_TEMPLATE.format(text=text.strip())
    out = classifier(
        prompt,
        max_new_tokens=8,
        return_full_text=False  # <<< KEY: don't echo the prompt
    )[0]["generated_text"].strip()

    # take only the first non-empty line
    first = out.splitlines()[0].strip()

    # normalize variants like "rant no visit", "Rant-No-Visit"
    norm = re.sub(r"[\s\-]+", "_", first.lower())

    mapping = {
        "advertisement": "Advertisement",
        "ad": "Advertisement",
        "ads": "Advertisement",
        "irrelevant": "Irrelevant",
        "rant_no_visit": "Rant_No_Visit",
        "valid": "Valid",
    }
    for k, v in mapping.items():
        if k == norm or k in norm:
            return v
    return "Valid"


def build_features(parquet_path, model_name="distilbert-base-uncased", max_len=128):
    # load file with cleaned reviews
    df = pd.read_parquet(parquet_path)

    # assign labels to each review
    df["label"] = df.apply(assign_label, axis=1)

    # fill in ambiguous cases with LLM prompt - if heuristics cannot classify, pass to LLM
    mask = df["label"].isna()
    if mask.any():
        df.loc[mask, "label"] = df.loc[mask, "text_clean"].apply(prompt_classify)

    # mapping - convert labels to integers or categorical ids
    df["label_id"] = df["label"].astype("category").cat.codes
    label2id = {label: idx for idx, label in enumerate(df["label"].astype("category").cat.categories)}
    id2label = {idx: label for label, idx in label2id.items()}

    # tokenizes review text using hugging face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(
        df["text_clean"].fillna("").tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )       

    labels = df["label_id"].tolist()
    return encodings, labels, df, label2id, id2label


if __name__ == "__main__":
    parquet_path = "data/processed/reviews_clean.parquet"  
    encodings, labels, df, label2id, id2label = build_features(parquet_path)

    print("Sample labels:", df[["text_clean", "label"]].head())
    print("Label map:", label2id)
