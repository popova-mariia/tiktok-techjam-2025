# Filtering the Noise: ML for Trustworthy Location Reviews

This repository contains code and resources for our hackathon project on **assessing the quality and relevancy of Google location reviews and alighing them with a set of policies** using Machine Learning and Natural Language Processing (NLP).

---

## Project Overview
Online reviews often contain **spam, advertisements, irrelevant content, or unhelpful rants**.  

Our project tackles this by:  
- **Filtering noisy reviews** with heuristic rules (ads, irrelevant topics, no-visit rants, low-quality).  
- **Building preprocessing pipelines** for CSV data.  
- **Extracting features** (linguistic, behavioral, and semantic).  
- **Training ML models** (scikit-learn + Transformers) to classify reviews as valid or invalid.  
- **Evaluating against policies** to enforce trustworthy review standards.  

This addresses **Prompt 1: Filtering the Noise – ML for Trustworthy Location Reviews**.

---

## Repository Structure

// to be updated

- `scripts/`
  - `features.py`
  - `model.py`
  - `policy.py`
  - `preprocess.py`
- `data/`   
  - `data/`          # raw input CSVs (ignored in git)
  - `processed/`    # preprocessed parquet + sample CSVs
- `README.md`
- `requirements.txt`
- `.gitignore`

---

## Preprocessing Script

### Features Extracted
The preprocessing step (`scripts/preprocess.py`) does the following:

- **HTML & URL cleaning**
  - Strips HTML tags
  - Removes URLs (flags if a URL was present)
- **Normalization**
  - Lowercasing
  - Whitespace collapsing
- **Signals & Features**
  - Review length (characters, tokens)
  - Count of exclamation marks, question marks
  - Count of ALLCAPS tokens
  - Boolean flag for very short reviews
- **Metadata**
  - Business name, author name
  - Rating (clipped to 1–5)
  - Category ID (mapped from raw rating category)
  - Has photo flag
  - Deterministic review ID (hash)
- **Deduplication**
  - Drops rows with empty text
  - Drops duplicates by `(business_name, author_name, text_clean)`
- **Outputs**
  - Saves `.parquet` file
  - Optional labelmap JSON (`category_id ↔ category_name`)
  - Sample `.csv` of first 2000 rows for inspection

### Usage

```bash
# Basic run
python scripts/preprocess.py \
  --input data/raw/google_reviews.csv \
  --output data/processed/reviews_clean.parquet

# With category label map
python scripts/preprocess.py \
  --input data/raw/google_reviews.csv \
  --output data/processed/reviews_clean.parquet \
  --labelmap_out data/processed/labelmap.json
