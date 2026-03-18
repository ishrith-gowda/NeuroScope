"""
federated averaging (fedavg) for cyclegan harmonization.

simulates federated training where each clinical site trains locally
on its own data, then shares model updates with a central server.
the server averages the updates and distributes the new global model.

this simulation uses a single machine with data partitioned by site
to mimic the distributed setting.

reference:
    mcmahan et al., "communication-efficient learning of deep networks
    from decentralized data", aistats 2017.
"""

import copy
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class FedAvgAggregator:
    """
    federated averaging aggregator.

    maintains a global model and coordinates local training across k clients.
    after each round, client model updates are averaged (weighted by dataset
    size) to produce the new global model.

    supports two modes:
        - share_all: average both generators and discriminators
        - share_generators_only: average only generators, keep discriminators
          local (recommended for gan training stability)
    """

    def __init__(
        self,
        global_model: nn.Module,
        share_discriminators: bool = False,
    ):
        """
        args:
            global_model: the global sa-cyclegan-2.5d model
            share_discriminators: if true, also average discriminator weights.
                                  if false, only average generator weights (recommended).
        """
        self.global_model = global_model
        self.share_discriminators = share_discriminators

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """get the current global model state dict."""
        if self.share_discriminators:
            return copy.deepcopy(self.global_model.state_dict())
        else:
            # only return generator parameters
            state = {}
            for k, v in self.global_model.state_dict().items():
                if "G_A2B" in k or "G_B2A" in k:
                    state[k] = v.clone()
            return state

    def distribute_to_clients(
        self, client_models: List[nn.Module]
    ) -> None:
        """
        distribute global model weights to all clients.

        args:
            client_models: list of client model instances
        """
        global_state = self.get_global_state()
        for client in client_models:
            client_state = client.state_dict()
            for k, v in global_state.items():
                if k in client_state:
                    client_state[k] = v.clone()
            client.load_state_dict(client_state)

    def aggregate(
        self,
        client_models: List[nn.Module],
        client_weights: Optional[List[float]] = None,
    ) -> None:
        """
        aggregate client model updates via weighted averaging.

        args:
            client_models: list of client models after local training
            client_weights: per-client weights (proportional to dataset size).
                           if none, uses uniform weights.
        """
        n_clients = len(client_models)
        if client_weights is None:
            client_weights = [1.0 / n_clients] * n_clients

        # normalize weights
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

        # get global state
        global_state = self.global_model.state_dict()
        aggregated_state = {}

        # determine which parameters to aggregate
        keys_to_aggregate = set()
        for k in global_state.keys():
            if self.share_discriminators:
                keys_to_aggregate.add(k)
            else:
                if "G_A2B" in k or "G_B2A" in k:
                    keys_to_aggregate.add(k)

        # weighted average
        for k in global_state.keys():
            if k in keys_to_aggregate:
                aggregated_state[k] = torch.zeros_like(global_state[k])
                for client, weight in zip(client_models, client_weights):
                    client_state = client.state_dict()
                    aggregated_state[k] += weight * client_state[k].to(
                        aggregated_state[k].device
                    )
            else:
                # keep global state for non-aggregated parameters
                aggregated_state[k] = global_state[k]

        self.global_model.load_state_dict(aggregated_state)

    def compute_model_divergence(
        self, client_models: List[nn.Module]
    ) -> Dict[str, float]:
        """
        compute divergence between client models and global model.

        useful for monitoring federated training stability. high divergence
        indicates clients are drifting apart.

        args:
            client_models: list of client models
        returns:
            dict with per-component divergence metrics
        """
        global_state = self.global_model.state_dict()
        divergences = {"G_A2B": 0.0, "G_B2A": 0.0}

        for client in client_models:
            client_state = client.state_dict()
            for k in global_state.keys():
                if "G_A2B" in k and "weight" in k:
                    diff = (client_state[k] - global_state[k]).norm().item()
                    divergences["G_A2B"] += diff
                elif "G_B2A" in k and "weight" in k:
                    diff = (client_state[k] - global_state[k]).norm().item()
                    divergences["G_B2A"] += diff

        n = len(client_models)
        for k in divergences:
            divergences[k] /= n

        return divergences


class FederatedSimulator:
    """
    end-to-end federated learning simulator for cyclegan.

    partitions data by site and simulates federated training rounds:
    1. server distributes global model to all clients
    2. each client trains locally for e epochs
    3. clients send updated models to server
    4. server aggregates updates via fedavg
    5. repeat for r communication rounds

    metrics tracked: convergence speed, final quality, communication cost,
    and model divergence across clients.
    """

    def __init__(
        self,
        model_factory,
        aggregator_class=FedAvgAggregator,
        n_clients: int = 2,
        local_epochs: int = 5,
        communication_rounds: int = 50,
        share_discriminators: bool = False,
    ):
        """
        args:
            model_factory: callable that creates a new sa-cyclegan-2.5d model
            aggregator_class: aggregation strategy class
            n_clients: number of federated clients (sites)
            local_epochs: epochs of local training per round
            communication_rounds: total number of federated rounds
            share_discriminators: whether to share discriminator weights
        """
        self.n_clients = n_clients
        self.local_epochs = local_epochs
        self.communication_rounds = communication_rounds

        # create global model and aggregator
        self.global_model = model_factory()
        self.aggregator = aggregator_class(
            self.global_model, share_discriminators=share_discriminators
        )

        # create client models (copies of global)
        self.client_models = [model_factory() for _ in range(n_clients)]

        # metrics history
        self.history = {
            "round": [],
            "global_metrics": [],
            "client_metrics": [],
            "divergence": [],
            "communication_bytes": [],
        }

    def estimate_communication_cost(self) -> int:
        """estimate bytes transmitted per round (model parameters * 4 bytes)."""
        n_params = sum(
            p.numel()
            for k, p in self.global_model.state_dict().items()
            if "G_A2B" in k or "G_B2A" in k
        )
        # upload from each client + download to each client
        bytes_per_round = n_params * 4 * self.n_clients * 2
        return bytes_per_round

    def run_round(
        self,
        client_trainers: list,
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        execute one federated round.

        args:
            client_trainers: list of trainer objects (one per client)
            client_weights: dataset size weights
        returns:
            round metrics
        """
        # distribute global model
        self.aggregator.distribute_to_clients(self.client_models)

        # local training
        client_metrics = []
        for i, (trainer, model) in enumerate(
            zip(client_trainers, self.client_models)
        ):
            trainer.model = model
            metrics = {}
            for _ in range(self.local_epochs):
                epoch_metrics = trainer.train_epoch(0)
                metrics = epoch_metrics
            client_metrics.append(metrics)

        # aggregate
        self.aggregator.aggregate(self.client_models, client_weights)

        # compute divergence
        divergence = self.aggregator.compute_model_divergence(self.client_models)

        return {
            "client_metrics": client_metrics,
            "divergence": divergence,
        }
