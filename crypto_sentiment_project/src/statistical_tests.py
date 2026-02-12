import numpy as np
from scipy import stats
import config

def get_metrics_distribution(df, date_list):
    """Extracts Panic (t-5) and Slope (t-14 to t-1)."""
    panic_metrics = []
    slope_metrics = []
    for date in date_list:
        try:
            if date not in df.index:
                continue
            loc = df.index.get_loc(date)
            #Metric A: Panic Lag (t = -5)
            if loc - 5 < 0:
                continue
            panic_metrics.append(df.iloc[loc - 5]['Panic_Index'])
            #Metric B: Slope
            start_loc = loc - 14
            if start_loc < 0:
                continue
            y = df.iloc[start_loc:loc]['Net_Sentiment'].values
            x = np.arange(len(y))
            if len(y) > 2:
                slope, _, _, _, _ = stats.linregress(x, y)
                slope_metrics.append(slope)
        except:
            continue
    return np.array(panic_metrics), np.array(slope_metrics)

def run_bootstrap_test(df, event_dates):
    """
    Runs Bootstrap Validation.
    """
    print(f"Running Bootstrap Validation ({config.BOOTSTRAP_ROUNDS} rounds):")
    np.random.seed(config.RANDOM_SEED)
    # Real Metrics
    real_panic, real_slope = get_metrics_distribution(df, event_dates)
    real_panic_mean = np.mean(real_panic)
    real_slope_mean = np.mean(real_slope)
    # Control Pool
    valid_indices = df.index[config.ANALYSIS_WINDOW:-config.ANALYSIS_WINDOW]
    control_dates = np.random.choice(valid_indices, size=1000, replace=True)
    ctrl_panic, ctrl_slope = get_metrics_distribution(df, control_dates)
    # Bootstrap Loop
    boot_panic_means = []
    boot_slope_means = []
    N = len(event_dates)
    for _ in range(config.BOOTSTRAP_ROUNDS):
        boot_panic_means.append(
            np.mean(np.random.choice(ctrl_panic, size=N, replace=True))
        )
        boot_slope_means.append(
            np.mean(np.random.choice(ctrl_slope, size=N, replace=True))
        )
    boot_panic_means = np.array(boot_panic_means)
    boot_slope_means = np.array(boot_slope_means)
    # P-Values
    p_panic = (np.sum(boot_panic_means >= real_panic_mean) + 1) / (config.BOOTSTRAP_ROUNDS + 1)
    p_slope = (np.sum(boot_slope_means <= real_slope_mean) + 1) / (config.BOOTSTRAP_ROUNDS + 1)
    # Confidence Intervals (95%)
    ci_panic_lower = np.percentile(boot_panic_means, 2.5)
    ci_panic_upper = np.percentile(boot_panic_means, 97.5)
    return {
        "real_panic_mean": real_panic_mean,
        "real_slope_mean": real_slope_mean,
        "p_panic": p_panic,
        "p_slope": p_slope,
        "ci_panic": (ci_panic_lower, ci_panic_upper)
    }

def compute_event_study_paths(df, event_dates):
    """
    Calculates the Mean Path and 95% Confidence Intervals for the Event Study.
    """
    window = config.ANALYSIS_WINDOW
    t_axis = np.arange(-window, window + 1)
    results = {'t': t_axis} 
    #Metrics to analyze
    metrics = ['close', 'Panic_Index', 'Net_Sentiment']
    #Helper to extract windows
    def get_windows(dates, col):
        windows = []
        for date in dates:
            if date not in df.index: continue
            loc = df.index.get_loc(date)
            if loc - window < 0 or loc + window >= len(df): continue
            data = df.iloc[loc - window : loc + window + 1][col].values
            if col == 'close':
                ref_price = data[window]
                if ref_price == 0: continue 
                data = data / ref_price - 1
            windows.append(data)
        return np.array(windows)
    np.random.seed(config.RANDOM_SEED)
    valid_range = df.index[window:-window]
    control_dates = np.random.choice(valid_range, size=len(event_dates)*3, replace=False)
    for metric in metrics:
        #1. Crash Group
        crash_data = get_windows(event_dates, metric)
        if len(crash_data) > 0:
            results[f'{metric}_crash_mean'] = np.mean(crash_data, axis=0)
            crash_se = stats.sem(crash_data, axis=0)
            results[f'{metric}_crash_ci_lower'] = results[f'{metric}_crash_mean'] - 1.96 * crash_se
            results[f'{metric}_crash_ci_upper'] = results[f'{metric}_crash_mean'] + 1.96 * crash_se
        else:
            results[f'{metric}_crash_mean'] = np.zeros(len(t_axis))
            results[f'{metric}_crash_ci_lower'] = np.zeros(len(t_axis))
            results[f'{metric}_crash_ci_upper'] = np.zeros(len(t_axis))
        #2. Control Group
        ctrl_data = get_windows(control_dates, metric)
        if len(ctrl_data) > 0:
            results[f'{metric}_ctrl_mean'] = np.mean(ctrl_data, axis=0)
            ctrl_se = stats.sem(ctrl_data, axis=0)
            results[f'{metric}_ctrl_ci_lower'] = results[f'{metric}_ctrl_mean'] - 1.96 * ctrl_se
            results[f'{metric}_ctrl_ci_upper'] = results[f'{metric}_ctrl_mean'] + 1.96 * ctrl_se
    return results
