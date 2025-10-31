import pandas as pd

df = pd.read_parquet("preprocessed/reviews.parquet")

# take 15 predicted off-topic
sample_off = df[df["flag_offtopic"] == 1].sample(22, random_state=0)
# take 15 predicted on-topic
#sample_on  = df[df["flag_offtopic"] == 0].sample(100, random_state=0)

sample = pd.concat([sample_off], ignore_index=True)
sample["label_human"] = ""   # <-- you fill this by hand: 1 = off-topic, 0 = ok

sample.to_csv("manual_check.csv", index=False)
print("Saved manual_check.csv")
