import sys
import os

script_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(script_path)
sys.path.insert(0, f"{current_file_dir}/../FinRL/")

import logging
import pandas as pd
import numpy as np
import datetime
import torch
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

from finrl.meta.preprocessor.preprocessors import GroupByScaler
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

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
 #    import pdb; pdb.set_trace()
   # df_t = df_t.set_index("date")
   fmt = "%Y-%m-%d"
   train_df = data_split(df, env.TRAIN_START.strftime(fmt), env.TRAIN_END.strftime(fmt))
   test_df = data_split(df, env.TEST_START.strftime(fmt), env.TEST_END.strftime(fmt))

   # setup portfolio optimization env
   # TODO: should we really group by scaler? Is there a better approach?
   environment = PortfolioOptimizationEnv(
           train_df,
           initial_amount=env.INITIAL_AMT,
           comission_fee_pct=env.COMISSION_FEE_PCT,
           time_window=env.TIME_WINDOW,
           features=env.FEATURES,
           normalize_df=None
    )

   model_kwargs = {
           "lr": env.LR,
           "policy": env.MODEL
    }

   policy_kwargs = {
           "k_size": 3,
           "time_window": 50
    }

   device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
   model = DRLAgent(environment).get_model("pg", device, model_kwargs, policy_kwargs)
   DRLAgent.train_model(model, episodes=40)

   torch.save(model.train_policy.state_dict(), "policy_EIIE.pt")
   import pdb; pdb.set_trace()




   

if __name__ == "__main__":
    main()
