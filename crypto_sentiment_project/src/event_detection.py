import pandas as pd
import config

def identify_crash_events(df):
    """
    Identifies crashes using Rolling Volatility Z-Score.
    """
    print("Detecting Crash Events (Regime-Adjusted):")
    df = df.copy()
    #Feature Engineering for Event Detection
    df['Return'] = df['close'].pct_change()
    df['Vol_30'] = df['Return'].rolling(window=config.VOLATILITY_WINDOW).std()
    df['Z_Return'] = df['Return'] / (df['Vol_30'] + 1e-6)
    #Candidates
    crash_candidates = df[df['Z_Return'] < config.CRASH_Z_THRESHOLD].index
    #Cooldown Logic
    filtered_events = []
    last_event = pd.Timestamp("1970-01-01") 
    for date in crash_candidates:
        if (date - last_event).days > config.COOLDOWN_DAYS:
            loc = df.index.get_loc(date)
            if loc > config.ANALYSIS_WINDOW and loc < len(df) - config.ANALYSIS_WINDOW:
                filtered_events.append(date)
                last_event = date
    print(f"Found {len(filtered_events)} valid crash events.")
    return pd.Index(filtered_events)