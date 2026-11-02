"""Builds sample_data.csv (a small random sample) from the full Airbnb CSV.

Run from the project folder:  python make_sample.py
"""
import pandas as pd

SOURCE = "GuestSatisfactionPrediction.csv"
OUTPUT = "sample_data.csv"
N_ROWS = 3000

# Long text / unused columns: the app drops these anyway, and removing them
# keeps the sample small enough for GitHub and Streamlit Cloud.
text_cols = [
    'listing_url', 'summary', 'space', 'description', 'neighborhood_overview',
    'notes', 'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url',
    'host_url', 'host_about', 'host_location', 'host_neighbourhood', 'street',
    'neighbourhood', 'market', 'smart_location', 'country_code', 'latitude',
    'longitude', 'first_review', 'last_review', 'host_name', 'city', 'state'
]

df = pd.read_csv(SOURCE, low_memory=False)
df = df.drop(columns=[c for c in text_cols if c in df.columns])
sample = df.sample(n=min(N_ROWS, len(df)), random_state=42)
sample.to_csv(OUTPUT, index=False)

print(f"Saved {len(sample)} rows to {OUTPUT}")