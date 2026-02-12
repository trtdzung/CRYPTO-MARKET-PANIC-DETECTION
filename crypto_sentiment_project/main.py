import pandas as pd
import numpy as np
import config
from src import data_loader, feature_engineering, event_detection, statistical_tests, backtest, visualization

def main():
    print("Start crypto sentiment pipeline")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print("=" * 50)
    np.random.seed(config.RANDOM_SEED)
    #1. Data Loading
    df_raw = data_loader.load_raw_data()
    #2. Feature Engineering
    df = feature_engineering.calculate_z_scores(df_raw)
    df = feature_engineering.define_signals_and_regimes(df)
    df.to_csv(config.PROCESSED_FILE)
    print(f"Processed data saved to {config.PROCESSED_FILE}")
    #3. Event Detection
    crash_events = event_detection.identify_crash_events(df)
    #4. Statistical Validation (Bootstrap)
    stats_results = statistical_tests.run_bootstrap_test(df, crash_events)
    print("\nStats report:")
    print(f"Panic Spike P-Value: {stats_results['p_panic']:.5f}")
    print(f"(95% CI: {stats_results['ci_panic'][0]:.2f} - {stats_results['ci_panic'][1]:.2f})")
    print(f"Slope P-Value: {stats_results['p_slope']:.5f}")
    print("\nGenerate Research-Grade Report:")
    #1. Price Path
    visualization.plot_price_regimes_log(df, crash_events)
    #2. Event Study (Compute -> Plot)
    print("Computute Event Study Paths:")
    event_data = statistical_tests.compute_event_study_paths(df, crash_events)
    visualization.plot_robust_event_study(event_data)
    #3. Forward Returns (Compute -> Plot)
    print("Computute Forward Returns:")
    sig_rets, ctrl_rets = backtest.compute_forward_returns(df)
    visualization.plot_forward_return_distribution(sig_rets, ctrl_rets)
    #4. Strategy Curve (Compute -> Plot)
    print("Computute Equity Curve:")
    equity_df = backtest.compute_cumulative_strategy(df)
    visualization.plot_cumulative_strategy(equity_df)
    #5. Regime Stats
    visualization.plot_regime_stats(df)

if __name__ == "__main__":
    main()
