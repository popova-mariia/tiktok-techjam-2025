#!/usr/bin/env python3
# preprocess.py
#
# Usage:
#   python src/preprocess.py --in data/translated_reviews.csv --outdir preprocessed --valid_frac 0.2
#
# Input columns expected:
#   business_name | author_name | text
#
# Outputs:
#   preprocessed/reviews.parquet       (full cleaned dataset + features)
#   preprocessed/train.parquet         (train split)
#   preprocessed/valid.parquet         (validation split)
#   preprocessed/report.txt            (quick distribution report)
#
# Notes:
# - No heavy deps. Install: pandas, numpy, regex (pip install pandas numpy regex)
# - You can extend weak-label heuristics as you iterate.

import argparse, os, re, math, json
import unicodedata
import numpy as np
import pandas as pd
import regex as rx

# ---------------------------------------------------------------------
# REGEXES
# ---------------------------------------------------------------------
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")
CODE_RE  = re.compile(r"`[^`]+`|``[^`]+``")
WS_RE    = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{4,}")  # e.g. loooool, !!!!!!
NONPRINT = ''.join(c for c in map(chr, range(256)) if unicodedata.category(c) in ('Cc', 'Cf'))
NONPRINT_TABLE = str.maketrans('', '', NONPRINT)

PROMO_KEYWORDS = [
    "dm me", "contact me", "call now", "subscribe", "follow me"
]
PROMO_RE = re.compile(r"|".join(map(re.escape, PROMO_KEYWORDS)), re.IGNORECASE)

REPEAT_LETTERS_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)

def collapse_repeats(text: str) -> str:
    """turn 'Awesomeee' -> 'Awesome', 'Deliciousss' -> 'Delicious'"""
    if not isinstance(text, str):
        return ""
    # replace 3+ of the same char with just 1
    return REPEAT_LETTERS_RE.sub(r"\1", text)


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\u200b", " ")  # zero-width space
    t = t.translate(NONPRINT_TABLE)  # remove control/non-printing
    t = unicodedata.normalize("NFKC", t)
    t = t.strip()
    t = CODE_RE.sub(" ", t)          # strip code blocks/backticks
    t = WS_RE.sub(" ", t)
    return t

def normalize_for_dupes(text: str) -> str:
    t = basic_clean(text).lower()
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = PHONE_RE.sub(" ", t)
    t = WS_RE.sub(" ", t)
    return t

def text_features(s: pd.Series) -> pd.DataFrame:
    # vectorized feature extraction
    text = s.fillna("")
    lens = text.str.len()
    n_words = text.str.split().str.len().fillna(0)

    urls = text.str.count(URL_RE)
    emails = text.str.count(EMAIL_RE)
    phones = text.str.count(PHONE_RE)

    exclam = text.str.count(r"!")
    ques   = text.str.count(r"\?")
    caps   = text.apply(lambda x: sum(1 for c in x if c.isupper()))
    letters = text.apply(lambda x: sum(1 for c in x if c.isalpha()))

    caps_ratio = (caps / (letters.replace(0, np.nan))).fillna(0.0)
    punct_total = exclam + ques
    punct_ratio = (punct_total / (lens.replace(0, np.nan))).fillna(0.0)

    repeats = text.str.count(REPEAT_RE)

    return pd.DataFrame({
        "len_chars": lens,
        "len_words": n_words,
        "num_urls": urls,
        "num_emails": emails,
        "num_phones": phones,
        "num_exclaim": exclam,
        "num_question": ques,
        "caps_ratio": caps_ratio,
        "punct_ratio": punct_ratio,
        "num_repeats": repeats,
    })


def weak_policy_flags(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["text_clean"].fillna("")
    # normalized for matching (handles Awesomeee, Deliciousss)
    txt_norm = txt.apply(collapse_repeats)

    # ------------------------------------------------------------
    # 0) very common opinion / review stubs (SHORT BUT VALID)
    # ------------------------------------------------------------
    # add simple -ly support for common sentiment words
    SHORT_OPINION_RE = re.compile(
        r"\b("
        # base adjectives
        r"nice|good|great|excellent|amazing|awesome|perfect|cool|beautiful|"
        r"delicious|tasty|yummy|fresh|classic|"
        r"recommend(ed|s)?|i recommend|we recommend|"
        r"like(d)|love(d)?|"
        r"ok|okay|it's ok|it's okay|its ok|not bad"
        r"the best|best\b|best in town|best place|best pub|"
        r"nothing interesting|"
        r"smiling employees?|friendly staff|"
         r"magnificent|wonderful|lovely|"
        r"thank you|thanks|grateful"
        r")"
        r"(ly)?"
        r"\b",
        re.IGNORECASE,
    )

    # if it's short AND matches an opinion → treat as reviewy
    short_valid = (
        (df["len_words"] <= 8)
        & txt_norm.str.contains(SHORT_OPINION_RE)
    )

    # ------------------------------------------------------------
    # price / value talk — often short but still reviews
    # ------------------------------------------------------------
    PRICE_RE = re.compile(
        r"("
        r"(cheap|affordable|budget|inexpensive)( place)?"
        r"|very cheap"
        r"|so cheap"
        r"|good price[s]?"
        r"|fair price[s]?"
        r"|(a bit|abit|kinda|quite|too)\s+(expensive|pricey)"
        r"|expensive place"
        r"|too expensive"
        r"|overpriced"
        r")",
        re.IGNORECASE,
    )
    price_valid = txt_norm.str.contains(PRICE_RE)

    # ------------------------------------------------------------
    # 1) ads / promos
    # ------------------------------------------------------------
    is_ad = (
        txt.str.contains(PROMO_RE)
        | (df["num_urls"] > 0)
        | (df["num_emails"] > 0)
        | (df["num_phones"] > 0)
    )

    # ------------------------------------------------------------
    # 2) review-ish gates (WHITELIST)
    # ------------------------------------------------------------
    VISIT_RE = re.compile(
        r"\b(i|we)\s+(went|visited|were there|came here|ordered|stayed|had (breakfast|lunch|dinner))\b",
        re.IGNORECASE,
    )
    REVIEWISH_RE = re.compile(
        r"\b("
        r"service|staff|employees?|food|menu|price|prices|"
        r"cheap|affordable|expensive|pricey|value|portion|"
        r"clean|location|ambiance|restaurant|cafe|hotel|room|delivery|order"
        r")\b",
        re.IGNORECASE,
    )
    OPINION_RE = re.compile(
        r"\b("
        r"good|great|excellent|amazing|awesome|nice|cool|"
        r"delicious(ly)?|tasty|"
        r"bad|terrible|awful|rude|slow|"
        r"overpriced|worth|recommended?"
        r")\b",
        re.IGNORECASE,
    )

    visit_like = txt_norm.str.contains(VISIT_RE)
    has_reviewish = txt_norm.str.contains(REVIEWISH_RE)
    has_opinion = txt_norm.str.contains(OPINION_RE)

    # final reviewy gate
    is_reviewy = short_valid | price_valid | visit_like | has_reviewish | has_opinion

    # ------------------------------------------------------------
    # 3) off-topic candidates (only if NOT reviewy)
    # ------------------------------------------------------------
    OFFTOPIC_RE = re.compile(
        r"\b(phone|iphone|samsung|laptop|crypto|bitcoin|nft|politics|election|stock market)\b",
        re.IGNORECASE,
    )
    generic_offtopic = txt_norm.str.contains(OFFTOPIC_RE)

    very_short = df["len_words"] <= 4
    short_chatter = very_short & (~is_reviewy)

    SLOGAN_RE = re.compile(
        r"(free palest|free ukraine|stand with|boycott|ceasefire now|black lives matter|blm\b)",
        re.IGNORECASE,
    )
    has_slogan = txt_norm.str.contains(SLOGAN_RE)

    mostly_links = (df["num_urls"] >= 2) & (df["len_words"] <= 10)
    noisy = (df["num_repeats"] >= 2)

    is_offtopic = (~is_reviewy) & ((df["len_words"] < 3) | (df["len_chars"] < 8)) | generic_offtopic | short_chatter | has_slogan | mostly_links | noisy

    # ------------------------------------------------------------
    # 4) rant no visit
    # ------------------------------------------------------------
    NO_VISIT_RE = re.compile(r"(never been|haven't been|didn't go|not visited)", re.IGNORECASE)
    is_rant_no_visit = txt_norm.str.contains(NO_VISIT_RE)

    # ------------------------------------------------------------
    # 5) invalid / junk
    # ------------------------------------------------------------

    return pd.DataFrame({
        "flag_ad_promo": is_ad.astype(int),
        "flag_offtopic": is_offtopic.astype(int),
        "flag_rant_no_visit": is_rant_no_visit.astype(int),
    })


def compute_sample_weight(row) -> float:
    # upweight rarer/flagged cases a bit to combat skew
    w = 1.0
    # each positive flag increases weight
    w += 0.5 * row["flag_ad_promo"]
    # made this gentler since we gated off-topic
    w += 0.4 * row["flag_offtopic"]
    w += 0.8 * row["flag_rant_no_visit"]
    # very short but not flagged? tiny downweight
    if row["len_words"] < 5 and (row["flag_ad_promo"] + row["flag_offtopic"] + row["flag_rant_no_visit"]) == 0:
        w *= 0.8
    return float(w)

def quick_report(df: pd.DataFrame) -> str:
    lines = []
    n = len(df)
    lines.append(f"# Rows: {n}")
    for c in ["flag_ad_promo", "flag_offtopic", "flag_rant_no_visit"]:
        if c in df.columns:
            lines.append(f"{c}: {df[c].sum()} ({df[c].mean():.3f})")
    lines.append(f"Deduped rows: {df['dedupe_rank'].eq(1).sum()} unique by business+text_norm")
    # length stats
    lines.append("len_words (p50/p90/p99): "
                 f"{df['len_words'].quantile(0.5):.0f}/"
                 f"{df['len_words'].quantile(0.9):.0f}/"
                 f"{df['len_words'].quantile(0.99):.0f}")
    return "\n".join(lines)

def train_valid_split(df: pd.DataFrame, valid_frac: float = 0.2, seed: int = 42):
    # stratify-ish by a composite flag to keep minority signals in both splits
    strat = (df["flag_ad_promo"]*4 + df["flag_offtopic"]*2 + df["flag_rant_no_visit"]).clip(0, 7)
    # group by business to reduce leakage (optional)
    rng = np.random.default_rng(seed)
    businesses = df["business_name"].fillna("_NA_").unique()
    rng.shuffle(businesses)
    cut = int(len(businesses) * (1 - valid_frac))
    train_biz = set(businesses[:cut])
    valid_biz = set(businesses[cut:])

    train = df[df["business_name"].fillna("_NA_").isin(train_biz)].copy()
    valid = df[df["business_name"].fillna("_NA_").isin(valid_biz)].copy()

    # if extremely unbalanced after split, just fallback to random split
    if min(len(train), len(valid)) < 50:
        msk = rng.random(len(df)) < (1 - valid_frac)
        train, valid = df[msk].copy(), df[~msk].copy()

    return train, valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV file (raw reviews)")
    ap.add_argument("--outdir", default="preprocessed", help="Output directory")
    ap.add_argument("--valid_frac", type=float, default=0.2, help="Validation fraction")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) load
    df = pd.read_csv(args.inp)
    expected = {"business_name", "author_name", "text"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2) drop obvious junk rows
    df = df.dropna(subset=["text"]).copy()
    df["text_clean"] = df["text"].apply(basic_clean)
    df = df[df["text_clean"].str.strip().ne("")].copy()

    # 3) near-duplicate removal (same business + same normalized text)
    df["text_norm"] = df["text_clean"].apply(normalize_for_dupes)
    df["dupe_key"] = df["business_name"].astype(str).str.strip().str.lower() + "||" + df["text_norm"]
    df["dedupe_rank"] = df.groupby("dupe_key")["dupe_key"].rank(method="first")
    df = df[df["dedupe_rank"] == 1].copy()

    # 4) features
    feats = text_features(df["text_clean"])
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    # 5) weak labels / policy flags (UPDATED)
    flags = weak_policy_flags(df)
    df = pd.concat([df.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)

    # 6) sample weights
    df["sample_weight"] = df.apply(compute_sample_weight, axis=1)

    # 7) save full dataset
    full_out = os.path.join(args.outdir, "reviews.parquet")
    df.to_parquet(full_out, index=False)

    # 8) split train/valid
    train, valid = train_valid_split(df, valid_frac=args.valid_frac)
    train_out = os.path.join(args.outdir, "train.parquet")
    valid_out = os.path.join(args.outdir, "valid.parquet")
    train.to_parquet(train_out, index=False)
    valid.to_parquet(valid_out, index=False)

    # 9) quick report
    rep = quick_report(df)
    with open(os.path.join(args.outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")

    print("=== Preprocess done ===")
    print(f"Full:   {full_out}  rows={len(df)}")
    print(f"Train:  {train_out} rows={len(train)}")
    print(f"Valid:  {valid_out} rows={len(valid)}")
    print("---")
    print(rep)

if __name__ == "__main__":
    main()
