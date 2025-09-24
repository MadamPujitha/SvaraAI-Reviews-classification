# src/preprocess.py
import re
import pandas as pd

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace('\r',' ').replace('\n',' ')
    s = re.sub(r'\S+@\S+', ' ', s)                  # remove emails
    s = re.sub(r'http\S+|www\.\S+', ' ', s)         # remove URLs
    s = re.sub(r'[^0-9A-Za-z\s\.,!?\'"]+', ' ', s)  # keep most punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df['reply'] = df['reply'].astype(str).apply(clean_text)
    df = df[df['reply'].str.len() > 0].copy()
    return df
