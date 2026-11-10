import pandas as pd
df = pd.read_csv("reviews_labeled.csv")
df = df[['review_text','sentiment','star','product_url']]
df.to_csv("reviews.csv")