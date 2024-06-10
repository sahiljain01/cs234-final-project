import torch
import numpy as np
import pandas as pd
import os

from torch_geometric.utils import k_hop_subgraph

from env_portfolio_optimization import PortfolioOptimizationEnv
from architectures import CustomGPM, GPM
from algorithms import TD3, PPO
from models import DRLAgent

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


#save model
model_save_dir = "models"
model_filename_TD3 = "TD3_GPM_.pt"
model_filename_PPO = "PPO_GPM_.pt"

metric_save_dir = "metrics"
metric_filename_TD3 = "TD3_GPM_.txt"
metric_filename_PPO = "PPO_GPM_.txt"

model_save_path_TD3 = os.path.join(model_save_dir, model_filename_TD3)
metric_save_path_TD3 = os.path.join(metric_save_dir, metric_filename_TD3)
model_save_path_PPO = os.path.join(model_save_dir, model_filename_PPO)
metric_save_path_PPO = os.path.join(metric_save_dir, metric_filename_PPO)

# Create the directory if it does not exist
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(metric_save_dir, exist_ok=True)

actor_critic = CustomGPM(new_edge_index, new_edge_type, nodes_to_select, device=device)
actor_critic_target = CustomGPM(new_edge_index, new_edge_type, nodes_to_select, device=device)

print(f"{device} is used")
#TD3_1
#model_td3 = TD3(environment_train, environment_test, model_save_path_TD3, metric_save_path_TD3, actor_critic, actor_critic_target,
#                device=device, lr=1e-4, buffer_size=5000, batch_size=64, target_policy_noise=0.1, target_noise_clip=0.2, policy_delay=3)
#TD3_2
#model_td3 = TD3(environment_train, environment_test, model_save_path_TD3, metric_save_path_TD3, actor_critic, actor_critic_target,
#               device=device, target_policy_noise=0.1, target_noise_clip=0.2, buffer_size=1000)
#TD3_default
#model_td3 = TD3(environment_train, environment_test, model_save_path_TD3, metric_save_path_TD3, actor_critic, actor_critic_target,
#                device=device)
#TD3_3
model_td3 = TD3(environment_train, environment_test, model_save_path_TD3, metric_save_path_TD3, actor_critic, actor_critic_target,
                device=device, lr=0.01, target_policy_noise=0.1)


print("start training TD3")
best_train_results_TD3, best_test_results_TD3 = model_td3.train(20)

GPM_results_TD3 = {
    "train": best_train_results_TD3,
    "test": best_test_results_TD3
}

#PPO
actor_critic = CustomGPM(new_edge_index, new_edge_type, nodes_to_select, device=device)
#ppo1
#model_ppo = PPO(environment_train, environment_test, model_save_path_PPO, metric_save_path_PPO, actor_critic,
#                device=device, num_episodes=20, lr=1e-4, ent_coef=0.001, ppo_epochs=10, minibatch_size=32, buffer_size=1000)
#ppo2
#model_ppo = PPO(environment_train, environment_test, model_save_path_PPO, metric_save_path_PPO, actor_critic,
#                device=device, num_episodes=20, ent_coef=0.001, buffer_size=400)
#ppo3
model_ppo = PPO(environment_train, environment_test, model_save_path_PPO, metric_save_path_PPO, actor_critic,
                device=device, num_episodes=20, lr=1e-2, ent_coef=0)
#ppo_default
#model_ppo = PPO(environment_train, environment_test, model_save_path_PPO, metric_save_path_PPO, actor_critic,
#                device=device, num_episodes=20)


print("start training PPO")
best_train_results_PPO, best_test_results_PPO = model_ppo.run()

GPM_results_PPO = {
    "train": best_train_results_PPO,
    "test": best_test_results_PPO
}

#default PG
# set PolicyGradient parameters
print("start training PG")
model_kwargs = {
    "lr": 0.01,
    "policy": GPM,
}

# here, we can set GPM's parameters
policy_kwargs = {
    "edge_index": new_edge_index,
    "edge_type": new_edge_type,
    "nodes_to_select": nodes_to_select
}

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
DRLAgent.train_model(model, episodes=20)
torch.save(model.train_policy.state_dict(), "models/PG_GPM.pt")

GPM_results_PG = {
    "train": environment_train._asset_memory["final"],
    "test": {},
}

# instantiate an architecture with the same arguments used in training
# and load with load_state_dict.
policy = GPM(new_edge_index, new_edge_type, nodes_to_select, device=device)
policy.load_state_dict(torch.load("models/PG_GPM.pt"))

# testing
DRLAgent.DRL_validation(model, environment_test, policy=policy)
GPM_results_PG["test"] = environment_test._asset_memory["final"]


#test uniform buy and hold (ubah)
print("start training UBAH")
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
plt.plot(GPM_results_TD3["train"], label="TD3_GPM_")
plt.plot(GPM_results_PPO["train"], label="PPO_GPM_")
plt.plot(GPM_results_PG["train"], label="PG_GPM_")

plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.title("Performance in training period")
plt.legend()
train_plot_path = os.path.join(save_dir, "performance_training_period.png")
plt.savefig(train_plot_path)
plt.show()

plt.figure()
plt.plot(UBAH_results["test"], label="Buy and Hold")
plt.plot(GPM_results_TD3["test"], label="TD3_GPM_")
plt.plot(GPM_results_PPO["test"], label="PPO_GPM_")
plt.plot(GPM_results_PG["test"], label="PG_GPM_")

plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.title("Performance in testing period")
plt.legend()
test_plot_path = os.path.join(save_dir, "performance_testing_period.png")
plt.savefig(test_plot_path)
plt.show()

print(f"Training plot saved to {train_plot_path}")
print(f"Testing plot saved to {test_plot_path}")
