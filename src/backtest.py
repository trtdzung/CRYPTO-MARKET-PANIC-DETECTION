import pandas as pd
import numpy as np
from scipy import stats
import config

def compute_cumulative_strategy(df):
    """
    Calculates the equity curve for the Contrarian Strategy vs Benchmark.
    """
    df = df.copy()
    df['Daily_Ret'] = df['close'].pct_change().fillna(0)
    signal = (df['Panic_Index'] > config.PANIC_THRESHOLD).astype(int)
    strategy_ret = signal.shift(1) * df['Daily_Ret']
    equity_df = pd.DataFrame(index=df.index)
    equity_df['Market'] = (1 + df['Daily_Ret']).cumprod()
    equity_df['Strategy'] = (1 + strategy_ret).cumprod()
    return equity_df

def compute_forward_returns(df):
    """
    Separates forward returns into Signal (Panic) and Control (Neutral).
    """
    df = df.copy()
    fwd_days = 5
    df['Fwd_Ret'] = df['close'].shift(-fwd_days) / df['close'] - 1
    signal_mask = df['Panic_Index'] > config.PANIC_THRESHOLD
    sig_rets = df.loc[signal_mask, 'Fwd_Ret'].dropna()
    ctrl_rets = df.loc[~signal_mask, 'Fwd_Ret'].dropna()
    return sig_rets, ctrl_rets

def analyze_lag_structure(df, event_dates):
    """
    Checks if Panic peaks before or after the crash.
    """
    print("Analyze Lag Structure:")
    lag_data = []
    for date in event_dates:
        try:
            loc = df.index.get_loc(date)
            if loc - 5 < 0 or loc + 6 >= len(df):
                continue
            window = df.iloc[loc-5 : loc+6]['Panic_Index'].values
            if len(window) == 11:
                lag_data.append(window)
        except:
            continue
    if not lag_data:
        return
    avg_path = np.mean(lag_data, axis=0)
    peak_idx = np.argmax(avg_path)
    peak_t = list(range(-5, 6))[peak_idx]
    print(f"Peak Panic occurs at t = {peak_t}")
    return avg_path

def run_contrarian_test(df):
    """
    Statistical T-Test for the 'Buy when Panic > Threshold' strategy.
    """
    print("Contrarian Strategy Test:")
    sig_ret, ctrl_ret = compute_forward_returns(df)
    
    if len(sig_ret) < 2 or len(ctrl_ret) < 2:
        print("Not enough data for T-Test.")
        return
    t_stat, p_val = stats.ttest_ind(sig_ret, ctrl_ret, equal_var=False)
    print(f"Mean Return (Signal): {sig_ret.mean()*100:.2f}%")
    print(f"Mean Return (Base):   {ctrl_ret.mean()*100:.2f}%")
    print(f"P-Value: {p_val:.5f}")
