from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import Sequential
from torch_geometric.utils import to_dense_batch


class EIIE(nn.Module):
    def __init__(
        self,
        initial_features=3,
        k_size=3,
        conv_mid_features=2,
        conv_final_features=20,
        time_window=50,
        device="cpu",
    ):
        """EIIE (ensemble of identical independent evaluators) policy network
        initializer.

        Args:
            initial_features: Number of input features.
            k_size: Size of first convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            time_window: Size of time window used as agent's state.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.48550/arXiv.1706.10059.
        """
        super().__init__()
        self.device = device

        n_size = time_window - k_size + 1

        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_size),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_size),
            ),
            nn.ReLU(),
        )

        self.final_convolution = nn.Conv2d(
            in_channels=conv_final_features + 1, out_channels=1, kernel_size=(1, 1)
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation.
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        output = self.sequential(observation)  # shape [N, 20, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [last_stocks, output], dim=1
        )  # shape [N, 21, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output)  # shape [N, 1, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [N, 1, PORTFOLIO_SIZE + 1, 1]

        # output shape must be [N, features] = [1, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: Environment observation (dictionary).
          last_action: Last action performed by the agent.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
            Last stocks and cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias


class EI3(nn.Module):
    def __init__(
        self,
        initial_features=3,
        k_short=3,
        k_medium=21,
        conv_mid_features=3,
        conv_final_features=20,
        time_window=50,
        device="cpu",
    ):
        """EI3 (ensemble of identical independent inception) policy network
        initializer.

        Args:
            initial_features: Number of input features.
            k_short: Size of short convolutional kernel.
            k_medium: Size of medium convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            time_window: Size of time window used as agent's state.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.1145/3357384.3357961.
        """
        super().__init__()
        self.device = device

        n_short = time_window - k_short + 1
        n_medium = time_window - k_medium + 1
        n_long = time_window

        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
            ),
            nn.ReLU(),
        )

        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=conv_mid_features, kernel_size=(1, k_medium)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
            ),
            nn.ReLU(),
        )

        self.long_term = nn.Sequential(nn.MaxPool2d(kernel_size=(1, n_long)), nn.ReLU())

        self.final_convolution = nn.Conv2d(
            in_channels=2 * conv_final_features + initial_features + 1,
            out_channels=1,
            kernel_size=(1, 1),
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation.
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        short_features = self.short_term(observation)
        medium_features = self.mid_term(observation)
        long_features = self.long_term(observation)

        features = torch.cat(
            [last_stocks, short_features, medium_features, long_features], dim=1
        )
        output = self.final_convolution(features)
        output = torch.cat([cash_bias, output], dim=2)

        # output shape must be [N, features] = [1, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: Environment observation (dictionary).
          last_action: Last action performed by the agent.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
            Last stocks and cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias



class GPM(nn.Module):
    def __init__(
        self,
        edge_index,
        edge_type,
        nodes_to_select,
        initial_features=3,
        k_short=3,
        k_medium=21,
        conv_mid_features=3,
        conv_final_features=20,
        graph_layers=1,
        time_window=50,
        softmax_temperature=1,
        device="cpu",
    ):
        """GPM (Graph-based Portfolio Management) policy network initializer.

        Args:
            edge_index: Graph connectivity in COO format.
            edge_type: Type of each edge in edge_index.
            nodes_to_select: ID of nodes to be selected to the portfolio.
            initial_features: Number of input features.
            k_short: Size of short convolutional kernel.
            k_medium: Size of medium convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            graph_layers: Number of graph neural network layers.
            time_window: Size of time window used as agent's state.
            softmax_temperature: Temperature parameter to softmax function.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.1016/j.neucom.2022.04.105.
        """
        super().__init__()
        self.device = device
        self.softmax_temperature = softmax_temperature

        num_relations = np.unique(edge_type).shape[0]

        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        self.edge_index = edge_index.to(self.device).long()

        if isinstance(edge_type, np.ndarray):
            edge_type = torch.from_numpy(edge_type)
        self.edge_type = edge_type.to(self.device).long()

        if isinstance(nodes_to_select, np.ndarray):
            nodes_to_select = torch.from_numpy(nodes_to_select)
        elif isinstance(nodes_to_select, list):
            nodes_to_select = torch.tensor(nodes_to_select)
        self.nodes_to_select = nodes_to_select.to(self.device)

        n_short = time_window - k_short + 1
        n_medium = time_window - k_medium + 1
        n_long = time_window

        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
            ),
            nn.ReLU(),
        )

        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features, out_channels=conv_mid_features, kernel_size=(1, k_medium)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
            ),
            nn.ReLU(),
        )

        self.long_term = nn.Sequential(nn.MaxPool2d(kernel_size=(1, n_long)), nn.ReLU())

        feature_size = 2 * conv_final_features + initial_features

        graph_layers_list = []
        for i in range(graph_layers):
            graph_layers_list += [
                (
                    RGCNConv(feature_size, feature_size, num_relations),
                    "x, edge_index, edge_type -> x",
                ),
                nn.LeakyReLU(),
            ]

        self.gcn = Sequential("x, edge_index, edge_type", graph_layers_list)

        self.final_convolution = nn.Conv2d(
            in_channels=2 * feature_size + 1,
            out_channels=1,
            kernel_size=(1, 1),
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation.
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        short_features = self.short_term(observation)
        medium_features = self.mid_term(observation)
        long_features = self.long_term(observation)

        temporal_features = torch.cat(
            [short_features, medium_features, long_features], dim=1
        )  # shape [N, feature_size, num_stocks, 1]

        # add features to graph
        graph_batch = self._create_graph_batch(temporal_features, self.edge_index)

        # set edge index for the batch
        edge_type = self._create_edge_type_for_batch(graph_batch, self.edge_type)

        # perform graph convolution
        graph_features = self.gcn(
            graph_batch.x, graph_batch.edge_index, edge_type
        )  # shape [N * num_stocks, feature_size]
        graph_features, _ = to_dense_batch(
            graph_features, graph_batch.batch
        )  # shape [N, num_stocks, feature_size]
        graph_features = torch.transpose(
            graph_features, 1, 2
        )  # shape [N, feature_size, num_stocks]
        graph_features = torch.unsqueeze(
            graph_features, 3
        )  # shape [N, feature_size, num_stocks, 1]
        graph_features = graph_features.to(self.device)

        # concatenate graph features and temporal features
        features = torch.cat(
            [temporal_features, graph_features], dim=1
        )  # shape [N, 2 * feature_size, num_stocks, 1]

        # perform selection and add last stocks
        features = torch.index_select(
            features, dim=2, index=self.nodes_to_select
        )  # shape [N, 2 * feature_size, portfolio_size, 1]
        features = torch.cat([last_stocks, features], dim=1)

        # final convolution
        output = self.final_convolution(features)  # shape [N, 1, portfolio_size, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [N, 1, portfolio_size + 1, 1]

        # output shape must be [N, portfolio_size + 1] = [1, portfolio_size + 1], being N batch size
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, portfolio_size + 1]

        output = self.softmax(output / self.softmax_temperature)

        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: Environment observation (dictionary).
          last_action: Last action performed by the agent.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
          Last stocks and cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias

    def _create_graph_batch(self, features, edge_index):
        """Create a batch of graphs with the features.

        Args:
          features: Tensor of shape [batch_size, feature_size, num_stocks, 1].
          edge_index: Graph connectivity in COO format.

        Returns:
          A batch of graphs with temporal features associated with each node.
        """
        batch_size = features.shape[0]
        graphs = []
        for i in range(batch_size):
            x = features[i, :, :, 0]  # shape [feature_size, num_stocks]
            x = torch.transpose(x, 0, 1)  # shape [num_stocks, feature_size]
            new_graph = Data(x=x, edge_index=edge_index).to(self.device)
            graphs.append(new_graph)
        return Batch.from_data_list(graphs)

    def _create_edge_type_for_batch(self, batch, edge_type):
        """Create the edge type tensor for a batch of graphs.

        Args:
          batch: Batch of graph data.
          edge_type: Original edge type tensor.

        Returns:
          Edge type tensor adapted for the batch.
        """
        batch_edge_type = torch.clone(edge_type).detach()
        for i in range(1, batch.batch_size):
            batch_edge_type = torch.cat(
                [batch_edge_type, torch.clone(edge_type).detach()]
            )
        return batch_edge_type


class CustomGPM(nn.Module):
    def __init__(
            self,
            edge_index,
            edge_type,
            nodes_to_select,
            initial_features=3,
            k_short=3,
            k_medium=21,
            conv_mid_features=3,
            conv_final_features=20,
            graph_layers=1,
            time_window=50,
            hidden_size=128,
            softmax_temperature=1,
            dropout_rate=0.5,
            device="cpu",
    ):
        """GPM (Graph-based Portfolio Management) policy network initializer.

        Args:
            edge_index: Graph connectivity in COO format.
            edge_type: Type of each edge in edge_index.
            nodes_to_select: ID of nodes to be selected to the portfolio.
            initial_features: Number of input features.
            k_short: Size of short convolutional kernel.
            k_medium: Size of medium convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            graph_layers: Number of graph neural network layers.
            time_window: Size of time window used as agent's state.
            softmax_temperature: Temperature parameter to softmax function.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.1016/j.neucom.2022.04.105.
        """
        super().__init__()
        self.device = device
        self.softmax_temperature = softmax_temperature

        num_relations = np.unique(edge_type).shape[0]

        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        self.edge_index = edge_index.to(self.device).long()

        if isinstance(edge_type, np.ndarray):
            edge_type = torch.from_numpy(edge_type)
        self.edge_type = edge_type.to(self.device).long()

        if isinstance(nodes_to_select, np.ndarray):
            nodes_to_select = torch.from_numpy(nodes_to_select)
        elif isinstance(nodes_to_select, list):
            nodes_to_select = torch.tensor(nodes_to_select)
        self.nodes_to_select = nodes_to_select.to(self.device)

        n_short = time_window - k_short + 1
        n_medium = time_window - k_medium + 1
        n_long = time_window

        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
            ),
            nn.BatchNorm2d(conv_mid_features),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
            ),
            nn.BatchNorm2d(conv_final_features),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
        )

        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features, out_channels=conv_mid_features, kernel_size=(1, k_medium)
            ),
            nn.BatchNorm2d(conv_mid_features),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
            ),
            nn.BatchNorm2d(conv_final_features),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
        )

        self.long_term = nn.Sequential(nn.MaxPool2d(kernel_size=(1, n_long)), nn.LeakyReLU())

        feature_size = 2 * conv_final_features + initial_features

        graph_layers_list = []
        for i in range(graph_layers):
            graph_layers_list += [
                (
                    RGCNConv(feature_size, feature_size, num_relations),
                    "x, edge_index, edge_type -> x",
                ),
                nn.BatchNorm1d(feature_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            ]

        self.gcn = Sequential("x, edge_index, edge_type", graph_layers_list)


        portfolio_size = self.nodes_to_select.shape[0]
        # Actor
        self.actor_conv = nn.Conv2d(in_channels=2 * feature_size + 1,
                                    out_channels=1,
                                    kernel_size=(1, 1))
        self.actor_final = nn.Sequential(
            nn.Linear(portfolio_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, portfolio_size + 1),
            nn.Softmax(dim=-1)  # Applying softmax over the last dimension
        )

        # Critic 1
        self.critic_conv1 = nn.Conv2d(in_channels=2 * feature_size + 1,
                                    out_channels=1,
                                    kernel_size=(1, 1))
        self.critic_final1 = nn.Sequential(
            nn.Linear(portfolio_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        #Critic 2
        self.critic_conv2 = nn.Conv2d(in_channels=2 * feature_size + 1,
                                      out_channels=1,
                                      kernel_size=(1, 1))
        self.critic_final2 = nn.Sequential(
            nn.Linear(portfolio_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        #Value
        self.value_conv = nn.Conv2d(in_channels=2 * feature_size,
                                      out_channels=1,
                                      kernel_size=(1, 1))
        self.value_final = nn.Sequential(
            nn.Linear(portfolio_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )



    def pf(self, observation):
        """portfolio features given input x.

        Args:
          observation: environment observation.

        Returns:
          portfolio features
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        short_features = self.short_term(observation)
        medium_features = self.mid_term(observation)
        long_features = self.long_term(observation)

        temporal_features = torch.cat(
            [short_features, medium_features, long_features], dim=1
        )  # shape [N, feature_size, num_stocks, 1]

        # add features to graph
        graph_batch = self._create_graph_batch(temporal_features, self.edge_index)

        # set edge index for the batch
        edge_type = self._create_edge_type_for_batch(graph_batch, self.edge_type)

        # perform graph convolution
        graph_features = self.gcn(
            graph_batch.x, graph_batch.edge_index, edge_type
        )  # shape [N * num_stocks, feature_size]
        graph_features, _ = to_dense_batch(
            graph_features, graph_batch.batch
        )  # shape [N, num_stocks, feature_size]
        graph_features = torch.transpose(
            graph_features, 1, 2
        )  # shape [N, feature_size, num_stocks]
        graph_features = torch.unsqueeze(
            graph_features, 3
        )  # shape [N, feature_size, num_stocks, 1]
        graph_features = graph_features.to(self.device)

        # concatenate graph features and temporal features
        features = torch.cat(
            [temporal_features, graph_features], dim=1
        )  # shape [N, 2 * feature_size, num_stocks, 1]

        # perform selection and add last stocks
        features = torch.index_select(
            features, dim=2, index=self.nodes_to_select
        )  # shape [N, 2 * feature_size, portfolio_size, 1]

        return features

    def forward(self, observation, action, mode): #mode = {"critic1", "critic2", "actor"}
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(self.device).float()

        stocks, cash_bias = self._process_last_action(action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        features = self.pf(observation) # shape [N, 2 * feature_size, portfolio_size, 1]

        if mode == "actor":
            features = torch.cat([stocks, features], dim=1)  # shape [N, 2 * feature_size + 1, portfolio_size, 1]
            output = self.actor_conv(features)  # shape [N, 1, portfolio_size, 1]
            output = torch.cat(
                [cash_bias, output], dim=2
            )  # shape [N, 1, portfolio_size + 1, 1]

            output = torch.squeeze(output, 3)
            output = torch.squeeze(output, 1)  # shape [N, portfolio_size + 1]

            output = self.actor_final(output)  # shape [N, portfolio_size + 1]

        elif mode == "critic1":
            features = torch.cat([stocks, features], dim=1)  # shape [N, 2 * feature_size + 1, portfolio_size, 1]
            output = self.critic_conv1(features)  # shape [N, 1, portfolio_size, 1]
            output = torch.cat(
                [cash_bias, output], dim=2
            )  # shape [N, 1, portfolio_size + 1, 1]

            output = torch.squeeze(output, 3)
            output = torch.squeeze(output, 1)  # shape [N, portfolio_size + 1]

            output = self.critic_final1(output)  # shape [N, portfolio_size + 1]

        elif mode == "critic2":
            features = torch.cat([stocks, features], dim=1)  # shape [N, 2 * feature_size + 1, portfolio_size, 1]
            output = self.critic_conv2(features)  # shape [N, 1, portfolio_size, 1]
            output = torch.cat(
                [cash_bias, output], dim=2
            )  # shape [N, 1, portfolio_size + 1, 1]

            output = torch.squeeze(output, 3)
            output = torch.squeeze(output, 1)  # shape [N, portfolio_size + 1]

            output = self.critic_final2(output)  # shape [N, portfolio_size + 1]

        else: #value
            output = self.value_conv(features)  # shape [N, 1, portfolio_size, 1]
            output = torch.cat(
                [cash_bias, output], dim=2
            )  # shape [N, 1, portfolio_size + 1, 1]

            output = torch.squeeze(output, 3)
            output = torch.squeeze(output, 1)  # shape [N, portfolio_size + 1]

            output = self.value_final(output)  # shape [N, portfolio_size + 1]

        return output

    def track_actor_parameters(self):
        pf_params = list(self.short_term.parameters()) + list(self.mid_term.parameters()) + list(
            self.long_term.parameters()) + list(self.gcn.parameters())
        actor_params = list(self.actor_conv.parameters()) + list(self.actor_final.parameters())
        return pf_params + actor_params

    def track_critic1_parameters(self):
        return list(self.critic_conv1.parameters()) + list(self.critic_final1.parameters())

    def track_critic2_parameters(self):
        return list(self.critic_conv2.parameters()) + list(self.critic_final2.parameters())

    def track_value_parameters(self):
        return list(self.value_conv.parameters()) + list(self.value_final.parameters())

    def _process_last_action(self, last_action):
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
          Last stocks and cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias

    def _create_graph_batch(self, features, edge_index):
        """Create a batch of graphs with the features.

        Args:
          features: Tensor of shape [batch_size, feature_size, num_stocks, 1].
          edge_index: Graph connectivity in COO format.

        Returns:
          A batch of graphs with temporal features associated with each node.
        """
        batch_size = features.shape[0]
        graphs = []
        for i in range(batch_size):
            x = features[i, :, :, 0]  # shape [feature_size, num_stocks]
            x = torch.transpose(x, 0, 1)  # shape [num_stocks, feature_size]
            new_graph = Data(x=x, edge_index=edge_index).to(self.device)
            graphs.append(new_graph)
        return Batch.from_data_list(graphs)

    def _create_edge_type_for_batch(self, batch, edge_type):
        """Create the edge type tensor for a batch of graphs.

        Args:
          batch: Batch of graph data.
          edge_type: Original edge type tensor.

        Returns:
          Edge type tensor adapted for the batch.
        """
        batch_edge_type = torch.clone(edge_type).detach()
        for i in range(1, batch.batch_size):
            batch_edge_type = torch.cat(
                [batch_edge_type, torch.clone(edge_type).detach()]
            )
        return batch_edge_type
class GPMPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            edge_index,
            edge_type,
            nodes_to_select,
            initial_features=3,
            k_short=3,
            k_medium=21,
            conv_mid_features=3,
            conv_final_features=20,
            graph_layers=1,
            time_window=50,
            softmax_temperature=1,
            custom_device="cpu",
            use_sde=False,
            **kwargs
    ):
        super(GPMPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            use_sde=use_sde,
            **kwargs
        )

        self.custom_device = custom_device
        self.softmax_temperature = softmax_temperature
        self.action_dim = self.action_space.shape[0]

        num_relations = np.unique(edge_type).shape[0]

        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        self.edge_index = edge_index.to(self.custom_device).long()

        if isinstance(edge_type, np.ndarray):
            edge_type = torch.from_numpy(edge_type)
        self.edge_type = edge_type.to(self.custom_device).long()

        if isinstance(nodes_to_select, np.ndarray):
            nodes_to_select = torch.from_numpy(nodes_to_select)
        elif isinstance(nodes_to_select, list):
            nodes_to_select = torch.tensor(nodes_to_select)
        self.nodes_to_select = nodes_to_select.to(self.custom_device)

        n_short = time_window - k_short + 1
        n_medium = time_window - k_medium + 1
        n_long = time_window

        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
            ),
            nn.ReLU(),
        )

        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_medium),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
            ),
            nn.ReLU(),
        )

        self.long_term = nn.Sequential(nn.MaxPool2d(kernel_size=(1, n_long)), nn.ReLU())

        feature_size = 2 * conv_final_features + initial_features

        graph_layers_list = []
        for i in range(graph_layers):
            graph_layers_list += [
                (
                    RGCNConv(feature_size, feature_size, num_relations),
                    "x, edge_index, edge_type -> x",
                ),
                nn.LeakyReLU(),
            ]

        self.gcn = Sequential("x, edge_index, edge_type", graph_layers_list)

        self.final_convolution = nn.Conv2d(
            in_channels=2 * feature_size + 1,
            out_channels=1,
            kernel_size=(1, 1),
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

        # Adjust the features_dim based on the dimensions of the output tensor
        self.features_dim = action_space.shape[0]  # portfolio_size + 1

        mlp_extractor = create_mlp(
            input_dim=self.features_dim,
            output_dim=64,  # MLP output dimension will be determined by the last layer
            net_arch=[64, 64],
            activation_fn=nn.ReLU
        )
        self.mlp_extractor = nn.Sequential(*mlp_extractor)

        self.action_net = nn.Linear(64, self.action_space.shape[0])
        self.value_net = nn.Linear(64, 1)

        self.log_std = nn.Parameter(torch.zeros(self.action_space.shape[0]), requires_grad=True)

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        action_distribution = DiagGaussianDistribution(self.action_dim)
        action_distribution.proba_distribution(mean_actions, self.log_std)
        return action_distribution

    def _get_latent(self, obs):
        if isinstance(obs, dict):
            observation = obs.get("state", None)
            last_action = obs.get("last_action", None)
        else:
            raise ValueError("Observation must be a dictionary with keys 'state' and 'last_action'.")

        if observation is None or last_action is None:
            raise ValueError("Observation dictionary must contain 'state' and 'last_action'.")

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.custom_device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.custom_device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.custom_device)

        short_features = self.short_term(observation)
        medium_features = self.mid_term(observation)
        long_features = self.long_term(observation)

        temporal_features = torch.cat(
            [short_features, medium_features, long_features], dim=1
        )  # shape [N, feature_size, num_stocks, 1]

        graph_batch = self._create_graph_batch(temporal_features, self.edge_index)
        edge_type = self._create_edge_type_for_batch(graph_batch, self.edge_type)

        graph_features = self.gcn(
            graph_batch.x, graph_batch.edge_index, edge_type
        )  # shape [N * num_stocks, feature_size]
        graph_features, _ = to_dense_batch(
            graph_features, graph_batch.batch
        )  # shape [N, num_stocks, feature_size]
        graph_features = torch.transpose(
            graph_features, 1, 2
        )  # shape [N, feature_size, num_stocks]
        graph_features = torch.unsqueeze(
            graph_features, 3
        )  # shape [N, feature_size, num_stocks, 1]
        graph_features = graph_features.to(self.custom_device)

        features = torch.cat(
            [temporal_features, graph_features], dim=1
        )  # shape [N, 2 * feature_size, num_stocks, 1]

        features = torch.index_select(
            features, dim=2, index=self.nodes_to_select
        )  # shape [N, 2 * feature_size, portfolio_size, 1]
        features = torch.cat([last_stocks, features], dim=1)

        output = self.final_convolution(features)  # shape [N, 1, portfolio_size, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [N, 1, portfolio_size + 1, 1]

        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, portfolio_size + 1]

        output = self.softmax(output / self.softmax_temperature)

        # Reshape the output tensor to match the expected input dimensions of the MLP extractor
        output = output.view(output.size(0), -1)

        latent_pi = self.mlp_extractor(output)
        return latent_pi

    def forward(self, obs, deterministic=False, **kwargs):
        latent_pi = self._get_latent(obs)
        action_distribution = self._get_action_dist_from_latent(latent_pi)
        actions = action_distribution.get_actions(deterministic=deterministic)
        log_prob = action_distribution.log_prob(actions)
        values = self.value_net(latent_pi)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        latent_pi = self._get_latent(observation)
        action_distribution = self._get_action_dist_from_latent(latent_pi)
        actions = action_distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions):
        latent_pi = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_pi)
        return values, log_prob, entropy

    def _process_last_action(self, last_action):
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias

    def _create_graph_batch(self, features, edge_index):
        batch_size = features.shape[0]
        graphs = []
        for i in range(batch_size):
            x = features[i, :, :, 0]  # shape [feature_size, num_stocks]
            x = torch.transpose(x, 0, 1)  # shape [num_stocks, feature_size]
            new_graph = Data(x=x, edge_index=edge_index).to(self.custom_device)
            graphs.append(new_graph)
        return Batch.from_data_list(graphs)

    def _create_edge_type_for_batch(self, batch, edge_type):
        batch_edge_type = torch.clone(edge_type).detach()
        for i in range(1, batch.batch_size):
            batch_edge_type = torch.cat(
                [batch_edge_type, torch.clone(edge_type).detach()]
            )
        return batch_edge_type
