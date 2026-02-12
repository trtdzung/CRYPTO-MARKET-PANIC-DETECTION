import os
#PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
INPUT_FILE = os.path.join(DATA_DIR, 'crypto_master_dataset.csv')
PROCESSED_FILE = os.path.join(DATA_DIR, 'daily_behavioral_signals.csv')

#REPRODUCIBILITY
RANDOM_SEED = 42

#ENGINEERING PARAMETERS
ROLLING_WINDOW = 30
SMA_WINDOW = 7
MIN_PERIODS = 20  

#REGIME PARAMETERS
PANIC_THRESHOLD = 2.0
EUPHORIA_PERCENTILE = 0.80

#CRASH DETECTION PARAMETERS
VOLATILITY_WINDOW = 30
CRASH_Z_THRESHOLD = -2.5
COOLDOWN_DAYS = 30

#VALIDATION PARAMETERS
BOOTSTRAP_ROUNDS = 10000
ANALYSIS_WINDOW = 14

#RAW DATA PATHS (For Preprocessing)
RAW_COMMENTS_FILE = os.path.join(DATA_DIR, 'reddit_2023_2025.csv')
RAW_PRICE_FILE = os.path.join(DATA_DIR, 'btc_price_2023_2025.csv')
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base" 
BATCH_SIZE = 32