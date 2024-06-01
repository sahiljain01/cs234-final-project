import os
import logging
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import src.env as env 



def load_data() -> pd.DataFrame:
    # download data
    stocks = env.STOCKS
    script_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(script_path)
    data_path = os.path.join(
           current_file_dir, 
           "data", 
           f"{env.TIME_RANGE_START}-{env.TIME_RANGE_START}-{'-'.join(env.STOCKS)}.pkl"
    )
    if not os.path.isfile(data_path): 
       df = YahooDownloader(
               start_date = env.TIME_RANGE_START,
               end_date = env.TIME_RANGE_END,
               ticker_list = config_tickers.DOW_30_TICKER
            ).fetch_data()
       df.to_pickle(data_path)
    else:
       df = pd.read_pickle(data_path)
    return df

def main():
   logging.info("Starting pipeline")
   df = load_data()

   # preprocess data, calculate set of technical indicators on the data
   fe = FeatureEngineer(
           use_technical_indicator=True,
           tech_indicator_list = env.INDICATORS
        )

   df_t = fe.preprocess_data(df)
   df_t = df_t.set_index("date")
   train_df = data_split(df_t, env.TRAIN_START, end.TRAIN_END)
   test_df = data_split(df_t, env.TEST_START, env.TEST_END)

   stock_dimension = len(df_t.tic.unique())

if __name__ == "__main__":
    main()
