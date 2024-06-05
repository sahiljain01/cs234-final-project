import torch
import numpy as np
import pandas as pd

from torch_geometric.utils import k_hop_subgraph

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import CustomGPM
from finrl.agents.portfolio_optimization.algorithms import TD3

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

nasdaq_temporal = pd.read_csv("Temporal_Relational_Stock_Ranking_FinRL/temporal_data/NASDAQ_temporal_data.csv")
nasdaq_edge_index = np.load("Temporal_Relational_Stock_Ranking_FinRL/relational_data/edge_indexes/NASDAQ_sector_industry_edge_index.npy")
nasdaq_edge_type = np.load("Temporal_Relational_Stock_Ranking_FinRL/relational_data/edge_types/NASDAQ_sector_industry_edge_type.npy")
list_of_stocks = nasdaq_temporal["tic"].unique().tolist()
tics_in_portfolio = ["AAPL", "CMCSA", "CSCO", "FB", "HBAN", "INTC", "MSFT", "MU", "NVDA", "QQQ", "XIV"]

portfolio_nodes = []
for tic in tics_in_portfolio:
    portfolio_nodes.append(list_of_stocks.index(tic))

nodes_kept, new_edge_index, nodes_to_select, edge_mask = k_hop_subgraph(
    torch.LongTensor(portfolio_nodes),
    2,
    torch.from_numpy(nasdaq_edge_index),
    relabel_nodes=True,
)

# reduce temporal data
nodes_kept = nodes_kept.tolist()
nasdaq_temporal["tic_id"], _ = pd.factorize(nasdaq_temporal["tic"], sort=True)
nasdaq_temporal = nasdaq_temporal[nasdaq_temporal["tic_id"].isin(nodes_kept)]
nasdaq_temporal = nasdaq_temporal.drop(columns="tic_id")

# reduce edge type
new_edge_type = torch.from_numpy(nasdaq_edge_type)[edge_mask]
_, new_edge_type = torch.unique(new_edge_type, return_inverse=True)

df_portfolio = nasdaq_temporal[["day", "tic", "close", "high", "low"]]

df_portfolio_train = df_portfolio[df_portfolio["day"] < 979]
df_portfolio_test = df_portfolio[df_portfolio["day"] >= 979]

environment_train = PortfolioOptimizationEnv(
    df_portfolio_train,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=50,
    features=["close", "high", "low"],
    time_column="day",
    normalize_df=None,
    tics_in_portfolio=tics_in_portfolio,
    return_last_action=True
)

environment_test = PortfolioOptimizationEnv(
    df_portfolio_test,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=50,
    features=["close", "high", "low"],
    time_column="day",
    normalize_df=None,
    tics_in_portfolio=tics_in_portfolio,
    return_last_action=True
)


actor_critic = CustomGPM(new_edge_index, new_edge_type, nodes_to_select)
actor_critic_target = CustomGPM(new_edge_index, new_edge_type, nodes_to_select)
model_td3 = TD3(environment_train, environment_test, actor_critic, actor_critic_target, batch_size=10)
print("start training")
model_td3.train(10)
'''
actor_critic = CustomGPM(new_edge_index, new_edge_type, nodes_to_select)
model_ppo = PPO(environment_train, environment_test, actor_critic, buffer_size=10, minibatch_size=5, num_episodes=1)
print("start training")
model_ppo.run()'''

#save model
save_dir = "models"
filename = "TD3_GPM_.pt"
save_path = os.path.join(save_dir, filename)

# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

torch.save(model_td3.actor_critic.state_dict(), save_path)
print(f"Model saved to {save_path}")

GPM_results = {
    "train": environment_train._asset_memory["final"],
    "test": environment_test._asset_memory["final"]
}

#test uniform buy and hold (ubah)
UBAH_results = {
    "train": {},
    "test": {},
}

PORTFOLIO_SIZE = len(tics_in_portfolio)

# train period
terminated = False
environment_train.reset()
while not terminated:
    action = [0] + [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE
    _, _, terminated, _ = environment_train.step(action)
UBAH_results["train"] = environment_train._asset_memory["final"]

# test period
terminated = False
environment_test.reset()
while not terminated:
    action = [0] + [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE
    _, _, terminated, _ = environment_test.step(action)
UBAH_results["test"] = environment_test._asset_memory["final"]

#saving plots
import matplotlib.pyplot as plt

save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

plt.plot(UBAH_results["train"], label="Buy and Hold")
plt.plot(GPM_results["train"], label="GPM")

plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.title("Performance in training period")
plt.legend()
train_plot_path = os.path.join(save_dir, "performance_training_period.png")
plt.savefig(train_plot_path)
plt.show()

plt.figure()
plt.plot(UBAH_results["test"], label="Buy and Hold")
plt.plot(GPM_results["test"], label="GPM")

plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.title("Performance in testing period")
plt.legend()
test_plot_path = os.path.join(save_dir, "performance_testing_period.png")
plt.savefig(test_plot_path)
plt.show()

print(f"Training plot saved to {train_plot_path}")
print(f"Testing plot saved to {test_plot_path}")
