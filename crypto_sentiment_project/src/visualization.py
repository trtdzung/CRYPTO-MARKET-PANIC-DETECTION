import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import config
import numpy as np 

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['lines.linewidth'] = 1.5

def plot_price_regimes_log(df, crash_events):
    """
    Plots Price (Log) with Regime Backgrounds.
    """
    df_plot = df.copy()
    df_plot = df_plot[df_plot['close'] > 0] 
    req_cols = ['close', 'Regime', 'Panic_Index']
    if not all(col in df_plot.columns for col in req_cols):
        print(f"Missing columns for Price Plot. Required: {req_cols}")
        return
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_yscale('log')
    ax.plot(df_plot.index, df_plot['close'], color='#333333', linewidth=1, label='BTC Price (Log)', alpha=0.8)
    y_ticks = [15000, 20000, 30000, 40000, 60000, 80000, 100000, 150000]
    current_min = df_plot['close'].min()
    current_max = df_plot['close'].max()
    visible_ticks = [t for t in y_ticks if t >= current_min * 0.8 and t <= current_max * 1.2]
    ax.set_yticks(visible_ticks)
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(mticker.NullFormatter())
    ax.fill_between(df_plot.index, 0, 1, where=df_plot['Regime'] == 'Panic', 
                    color='#ff9999', alpha=0.3, label='Panic Regime', 
                    transform=ax.get_xaxis_transform())
    signals = df_plot[df_plot['Panic_Index'] > config.PANIC_THRESHOLD]
    ax.scatter(signals.index, signals['close'], color='green', marker='^', s=30, label='Contrarian Signal', zorder=3)
    valid_crashes = [d for d in crash_events if d in df_plot.index]
    if valid_crashes:
        crash_prices = df_plot.loc[valid_crashes]['close']
        ax.scatter(valid_crashes, crash_prices, color='black', marker='x', s=100, label='Crash Event', zorder=4) 
    ax.set_title('Market Microstructure: Price, Regimes & Signals', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (Log Scale, USD)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, '01_price_log_regimes.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return fig

def plot_robust_event_study(event_data):
    """
    Plots Pre-Computed Event Study Paths.
    """
    metrics = [
        ('close', 'Price Trajectory (Normalized)', '% Change'),
        ('Panic_Index', 'Panic Index Structure', 'Z-Score'),
        ('Net_Sentiment', 'Net Sentiment Structure', 'Score')
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    t = event_data['t']
    for i, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        mean_key = f'{metric}_crash_mean'
        if mean_key in event_data:
            ax.plot(t, event_data[mean_key], color='red', linewidth=2, label='Crash Event')
            ax.fill_between(
                t,
                event_data[f'{metric}_crash_ci_lower'],
                event_data[f'{metric}_crash_ci_upper'],
                color='red',
                alpha=0.15
            )
        mean_key_ctrl = f'{metric}_ctrl_mean'
        if mean_key_ctrl in event_data:
            ax.plot(t, event_data[mean_key_ctrl], color='gray', linestyle='--', linewidth=1.5, label='Random Baseline')
            ax.fill_between(
                t,
                event_data[f'{metric}_ctrl_ci_lower'],
                event_data[f'{metric}_ctrl_ci_upper'],
                color='gray',
                alpha=0.1
            )
        ax.axvline(0, color='black', linestyle=':', alpha=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend()
    axes[2].set_xlabel('Days Relative to Event (t=0)')
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, '02_robust_event_study.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return fig

def plot_forward_return_distribution(sig_rets, ctrl_rets):
    """
    Plots Distribution of Forward Returns.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
    s_r = sig_rets * 100
    c_r = ctrl_rets * 100
    sns.kdeplot(data=s_r, ax=axes[0], fill=True, color='green', label='Signal (Panic > 2.0)', alpha=0.3)
    sns.kdeplot(data=c_r, ax=axes[0], fill=True, color='gray', label='Control (Neutral)', alpha=0.1)
    axes[0].axvline(s_r.mean(), color='green', linestyle='--')
    axes[0].axvline(c_r.mean(), color='gray', linestyle='--')
    axes[0].set_title('Forward 5-Day Return Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[1].boxplot(
        [s_r, c_r],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor='#d9f0a3'),
        medianprops=dict(color='green')
    )
    axes[1].set_yticklabels(['Signal', 'Control'])
    axes[1].set_xlabel('5-Day Forward Return (%)')
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, '03_forward_return_dist.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return fig

def plot_cumulative_strategy(equity_df):
    """
    Plots Cumulative Equity Curve.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_df.index, equity_df['Market'], color='gray', label='Buy & Hold', alpha=0.6)
    ax.plot(equity_df.index, equity_df['Strategy'], color='green', label='Contrarian Panic Strategy', linewidth=2)
    mkt_perf = (equity_df['Market'].iloc[-1] - 1) * 100
    strat_perf = (equity_df['Strategy'].iloc[-1] - 1) * 100
    text_str = f"Strategy: {strat_perf:.1f}%\nBenchmark: {mkt_perf:.1f}%"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(
        0.02,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=props
    )
    ax.set_title('Cumulative Performance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend() 
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, '04_cumulative_strategy.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return fig

def plot_regime_stats(df):
    """
    Plots Regime Frequency.
    """
    if 'Regime' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df['Regime'].value_counts()
    colors = {
        'Neutral': 'gray',
        'Euphoria': 'green',
        'Panic': 'red',
        'Capitulation': 'black',
        'Distribution': 'orange'
    }
    bar_colors = [colors.get(x, 'blue') for x in counts.index]
    counts.plot(kind='bar', color=bar_colors, ax=ax)
    ax.set_title('Market Regime Frequency', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Days')
    total = len(df)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + 5
        ax.annotate(percentage, (x, y), ha='center')
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, '05_regime_stats.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return fig
