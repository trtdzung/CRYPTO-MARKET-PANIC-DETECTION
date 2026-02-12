import pandas as pd
import config

def load_raw_data():
    """
    Loads raw data with strict integrity checks.
    """
    print(f"Loading data from {config.INPUT_FILE}")
    try:
        df = pd.read_csv(config.INPUT_FILE)
        if 'date' not in df.columns:
            raise ValueError("Dataset missing required 'date' column.")
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.tz_localize(None)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        if df.index.has_duplicates:
            raise ValueError(f"Data contains {df.index.duplicated().sum()} duplicate dates.")
        if not df.index.is_monotonic_increasing:
            raise ValueError("Data is not sorted by date.")
        if df['close'].isnull().any():
            print("Missing 'close' prices found.")
            df['close'] = df['close'].ffill()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found at {config.INPUT_FILE}")
