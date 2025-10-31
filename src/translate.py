import pandas as pd
from langdetect import detect, DetectorFactory
from time import sleep

# Choose ONE translator:
# Option A: deep-translator (uses Google Translate web)
#   pip install deep-translator
from deep_translator import GoogleTranslator

# Reproducible language detection
DetectorFactory.seed = 0

in_path  = "data/reviews_combined.csv"   # <- set to your downloaded CSV
out_path = "data/translated_reviews.csv"        # <- this will feed preprocess.py

def safe_detect(text: str) -> str:
    try:
        t = (text or "").strip()
        if not t:
            return "und"
        return detect(t)
    except Exception:
        return "und"

def translate_to_en(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    # quick throttle to be nice
    sleep(0.15)
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text  # fallback: original if translation fails

def main():
    # 1) load your Apify CSV
    df = pd.read_csv(in_path, encoding="utf-8")

    # 2) rename to match preprocess.py
    df = df.rename(columns={
        "title": "business_name",
        "name": "author_name",
        "text": "text_original",
    })

    # keep only what we need + a few helpers
    cols = ["business_name", "author_name", "text_original"]
    df = df[cols].copy()

    # 3) detect language
    df["lang"] = df["text_original"].apply(safe_detect)

    # 4) translate only non-English
    df["text_en"] = df.apply(
        lambda r: r["text_original"] if r["lang"].startswith("en") else translate_to_en(r["text_original"]),
        axis=1,
    )

    # 5) produce the final three columns for your pipeline
    out = df.rename(columns={"text_en": "text"})[["business_name", "author_name", "text"]]

    # 6) save
    out.to_csv(out_path, index=False, encoding="utf-8")

    # quick stats
    print("Saved:", out_path)
    print("Rows:", len(out))
    print("Lang counts:\n", df["lang"].value_counts())

if __name__ == "__main__":
    main()
