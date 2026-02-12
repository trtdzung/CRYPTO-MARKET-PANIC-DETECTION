import pandas as pd
import numpy as np
import config

def calculate_z_scores(df):
    """
    Math Layer: Normalizes raw sentiment.
    """
    print("Executing Math Layer (Normalization):")
    metrics = ['Euphoria', 'FUD', 'Anger', 'Confusion']
    for m in metrics:
        col = f"{m}_SMA7" if f"{m}_SMA7" in df.columns else m
        rolling_mean = df[col].rolling(
            window=config.ROLLING_WINDOW,
            min_periods=config.MIN_PERIODS
        ).mean()
        rolling_std = df[col].rolling(
            window=config.ROLLING_WINDOW,
            min_periods=config.MIN_PERIODS
        ).std()
        df[f'{m}_Z'] = (df[col] - rolling_mean) / (rolling_std + 1e-6)
    return df

def define_signals_and_regimes(df):
    """
    Logic Layer: Defines Composite Signals and Market Regimes.
    """
    print("Executing Logic Layer (Signal Generation):")
    #1. Composite Signals
    e_col = 'Euphoria_SMA7' if 'Euphoria_SMA7' in df.columns else 'Euphoria'
    f_col = 'FUD_SMA7' if 'FUD_SMA7' in df.columns else 'FUD'
    df['Net_Sentiment'] = df[e_col] - df[f_col]
    #Panic Index
    a_col = 'Anger_SMA7' if 'Anger_SMA7' in df.columns else 'Anger'
    raw_panic = df[f_col] + df[a_col] 
    panic_mean = raw_panic.rolling(
        window=config.ROLLING_WINDOW,
        min_periods=config.MIN_PERIODS
    ).mean()
    panic_std = raw_panic.rolling(
        window=config.ROLLING_WINDOW,
        min_periods=config.MIN_PERIODS
    ).std()
    df['Panic_Index'] = (raw_panic - panic_mean) / (panic_std + 1e-6)
    
    #2. Regime Definition
    #Percentile calculated using expanding window to avoid lookahead bias
    df['Net_Sentiment_Percentile'] = (
        df['Net_Sentiment']
        .expanding(min_periods=config.MIN_PERIODS)
        .rank(pct=True)
    )
    df['Regime'] = 'Neutral'
    df.loc[
        df['Net_Sentiment_Percentile'] > config.EUPHORIA_PERCENTILE,
        'Regime'
    ] = 'Euphoria'
    df.loc[
        df['Panic_Index'] > config.PANIC_THRESHOLD,
        'Regime'
    ] = 'Panic'
    df.loc[
        df['Panic_Index'] > 3.0,
        'Regime'
    ] = 'Capitulation'
    return df.dropna()
