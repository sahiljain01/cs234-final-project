import datetime
from finrl import config_tickers
from finrl.config import INDICATORS

TIME_RANGE_START = datetime.datetime(2010, 1, 1)
TIME_RANGE_END = datetime.datetime(2020, 1, 1)

TEST_PERIOD = datetime.timedelta(days=3)


STOCKS = config_tickers.DOW_30_TICKER

INDICATORS = INDICATORS
