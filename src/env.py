import datetime
from finrl import config_tickers
from finrl.config import INDICATORS

TIME_RANGE_START = datetime.datetime(2010, 1, 1)
TIME_RANGE_END = datetime.datetime(2020, 1, 1)

TEST_PERIOD = datetime.timedelta(days=365)

TRAIN_START = TIME_RANGE_START
TRAIN_END = TIME_RANGE_END - TEST_PERIOD

TEST_START = TIME_RANGE_END - TEST_PERIOD
TEST_END = TIME_RANGE_END

STOCKS = config_tickers.DOW_30_TICKER

INDICATORS = INDICATORS
