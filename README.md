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
