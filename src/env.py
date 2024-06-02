import datetime
from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.agents.portfolio_optimization.architectures import EIIE


"""
SETUP TRAIN/TEST TIME RANGES
"""
TIME_RANGE_START = datetime.datetime(2010, 1, 1)
TIME_RANGE_END = datetime.datetime(2020, 1, 1)
VALIDATION_PERIOD = datetime.timedelta(days=365)
TEST_PERIOD = datetime.timedelta(days=365)

TRAIN_START = TIME_RANGE_START
TRAIN_END = TIME_RANGE_END - TEST_PERIOD - VALIDATION_PERIOD
VALIDATION_START = TRAIN_END
VALIDATION_END = VALIDATION_START + VALIDATION_PERIOD
TEST_START = VALIDATION_END
TEST_END = TIME_RANGE_END

"""
STOCKS AND INDICATORS TO USE
"""
STOCKS = config_tickers.DOW_30_TICKER
STOCKS_EXCLUDE = ["WBA", "DOW"] # exclude WBA bc/ no graph data, DOW bc missing data?
STOCKS = list(set(STOCKS) - set(STOCKS_EXCLUDE))
INDICATORS = INDICATORS

"""
ENVIRONMENT SETUP
"""
INITIAL_AMT = 100_000
COMISSION_FEE_PCT = 0
TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]

"""
MODEL SPECIFICS
"""
LR = 0.01
MODEL = EIIE





