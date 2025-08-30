import pandas as pd
from transformers import AutoTokenizer, pipeline

def assign_label(row):
    if row["looks_promo_heur"]:
        return "Advertisement"
    elif row["looks_rant_no_visit_heur"]:
        return "Rant_No_Visit"
    else:
        return None

PROMPT_TEMPLATE = """Classify the following review into exactly one of these categories:
- Irrelevant
- Rant_No_Visit
- Valid

Review: "{text}"
Answer:"""

classifier = pipeline("text-generation", model="Qwen/Qwen2-7B-Instruct")

def prompt_classify(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Low_Quality"

    prompt = PROMPT_TEMPLATE.format(text=text)
    result = classifier(prompt, max_new_tokens=10)[0]["generated_text"]

    # naive cleanup: pick the first matching label
    for label in ["Irrelevant", "Rant_No_Visit", "Valid"]:
        if label in result:
            return label
    return "Valid"  # fallback

def build_features(parquet_path, model_name="distilbert-base-uncased", max_len=128):
    # load file with cleaned reviews
    df = pd.read_parquet(parquet_path)

    # assign labels to each review
    df["label"] = df.apply(assign_label, axis=1)

    # fill in ambiguous cases with LLM prompt
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
