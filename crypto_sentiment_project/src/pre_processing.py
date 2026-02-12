import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
import config
import os

class TextCleaner:
    """Handles regex-based text cleaning."""
    @staticmethod
    def clean(text):
        if not isinstance(text, str): return ""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove Reddit handles
        text = re.sub(r'u/[a-zA-Z0-9_-]+', '', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
        return text.lower().strip()
    
class SentimentEngine:
    """
    Wraps the HuggingFace emotion model.
    Maps: Joy -> Euphoria, Fear -> FUD, Anger -> Anger.
    """
    def __init__(self):
        print(f"Loading model: {config.MODEL_NAME}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {self.device.upper()}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME).to(self.device)
        self.labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    def predict_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = softmax(outputs.logits, dim=1).cpu().numpy()
        return scores

def run_preprocessing_pipeline():
    print("Start upstream processing pipeline")
    #1. Load raw data
    try:
        df = pd.read_csv(config.RAW_COMMENTS_FILE)
        print(f"Loaded {len(df)} raw comments.")
    except FileNotFoundError:
        print(f"Error: Raw file not found at {config.RAW_COMMENTS_FILE}")
        return
    # 2.Clean text
    print("Cleaning text:")
    cleaner = TextCleaner()
    if 'body' not in df.columns:
        return
    df['clean_text'] = df['body'].apply(cleaner.clean)
    df = df[df['clean_text'].str.len() > 10]
    #3. Batch inference
    engine = SentimentEngine()
    print("Running inference:")
    emotions = {label: [] for label in engine.labels}
    batch_size = config.BATCH_SIZE
    texts = df['clean_text'].tolist()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        scores = engine.predict_batch(batch_texts)
        for j, label in enumerate(engine.labels):
            emotions[label].extend(scores[:, j])
    for label in engine.labels:
        df[label] = emotions[label]
    # 4. Aggregation (daily resampling)
    print("Aggregating to daily frequency:")
    if 'created_utc' in df.columns:
        df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        print("Error: No timestamp column found (created_utc or date).")
        return
    df.set_index('date', inplace=True)
    daily_df = df[engine.labels].resample('D').mean()
    daily_df.rename(columns={
        'joy': 'Euphoria',
        'fear': 'FUD',
        'anger': 'Anger',
        'surprise': 'Confusion'
    }, inplace=True)
    # 5.Merge with price data
    print(f"Merging with Price Data from {config.RAW_PRICE_FILE}...")
    try:
        price_df = pd.read_csv(config.RAW_PRICE_FILE)
        date_col = None
        for col in price_df.columns:
            if col.lower() in ['date', 'timestamp', 'snapped_at']:
                date_col = col
                break
        if date_col is None:
            raise ValueError(f"Could not find a date column in {config.RAW_PRICE_FILE}")
        price_df[date_col] = pd.to_datetime(price_df[date_col])
        if price_df[date_col].dt.tz is not None:
             price_df[date_col] = price_df[date_col].dt.tz_localize(None)
        price_df.set_index(date_col, inplace=True)
        price_df.index.name = 'date' 
        master_df = daily_df.join(price_df, how='inner')
        if master_df.empty:
            print("Merge failed: No overlapping dates found between Sentiment and Price.")
            print(f"Sentiment Range: {daily_df.index.min()} to {daily_df.index.max()}")
            print(f"Price Range: {price_df.index.min()} to {price_df.index.max()}")
            return
        print(f"Merged Data Shape: {master_df.shape}")
        #6. Save Final Master Dataset
        output_path = config.INPUT_FILE
        master_df.to_csv(output_path)
        print(f"Master Dataset saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: Price file not found at {config.RAW_PRICE_FILE}")
        return

if __name__ == "__main__":
    run_preprocessing_pipeline()