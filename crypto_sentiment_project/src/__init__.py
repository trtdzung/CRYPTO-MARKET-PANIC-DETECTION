#1. Data & Logic
from .data_loader import load_raw_data
from .feature_engineering import calculate_z_scores, define_signals_and_regimes
from .event_detection import identify_crash_events

#2. Analysis and Math
from .statistical_tests import run_bootstrap_test, compute_event_study_paths
from .backtest import (
    compute_forward_returns, 
    compute_cumulative_strategy, 
    analyze_lag_structure, 
    run_contrarian_test
)

#3. Visualization
from .visualization import (
    plot_price_regimes_log, 
    plot_robust_event_study, 
    plot_forward_return_distribution, 
    plot_cumulative_strategy, 
    plot_regime_stats
)

__all__ = [
    'load_raw_data',
    'calculate_z_scores',
    'define_signals_and_regimes',
    'identify_crash_events',
    'run_bootstrap_test',
    'compute_event_study_paths',
    'compute_forward_returns',
    'compute_cumulative_strategy',
    'analyze_lag_structure',
    'run_contrarian_test',
    'plot_price_regimes_log',
    'plot_robust_event_study',
    'plot_forward_return_distribution',
    'plot_cumulative_strategy',
    'plot_regime_stats'
]