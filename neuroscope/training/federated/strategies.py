"""
federated learning strategies beyond basic fedavg.

implements fedprox (li et al., mlsys 2020) which adds a proximal
regularization term to prevent client models from drifting too far
from the global model during local training.

reference:
    li et al., "federated optimization in heterogeneous networks",
    mlsys 2020.
"""

import copy
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from neuroscope.training.federated.fedavg import FedAvgAggregator


class FedProxAggregator(FedAvgAggregator):
    """
    fedprox aggregator with proximal regularization.

    extends fedavg by adding a proximal term to each client's local loss:
        L_local = L_original + (mu / 2) * ||w - w_global||^2

    this penalizes client models for deviating too far from the global
    model, which improves convergence when data is heterogeneous across
    sites (common in multi-center mri studies).
    """

    def __init__(
        self,
        global_model: nn.Module,
        mu: float = 0.01,
        share_discriminators: bool = False,
    ):
        """
        args:
            global_model: the global model
            mu: proximal regularization strength. higher values = less drift.
                recommended range: [0.001, 0.1]
            share_discriminators: whether to share discriminator weights
        """
        super().__init__(global_model, share_discriminators)
        self.mu = mu
        self._global_params_snapshot = None

    def snapshot_global_params(self) -> None:
        """
        take a snapshot of the global model parameters before local training.
        called before distributing to clients each round.
        """
        self._global_params_snapshot = {
            k: v.clone().detach()
            for k, v in self.global_model.state_dict().items()
        }

    def compute_proximal_loss(self, client_model: nn.Module) -> torch.Tensor:
        """
        compute the proximal regularization term for a client model.

        args:
            client_model: the client's local model after some local updates
        returns:
            proximal loss: (mu / 2) * sum(||w_local - w_global||^2)
        """
        if self._global_params_snapshot is None:
            raise RuntimeError(
                "must call snapshot_global_params() before computing proximal loss"
            )

        prox_loss = torch.tensor(0.0, device=next(client_model.parameters()).device)
        for k, local_param in client_model.named_parameters():
            if k in self._global_params_snapshot:
                global_param = self._global_params_snapshot[k].to(local_param.device)
                prox_loss += (local_param - global_param).pow(2).sum()

        return (self.mu / 2) * prox_loss

    def distribute_to_clients(self, client_models: List[nn.Module]) -> None:
        """distribute global model and take snapshot for proximal term."""
        self.snapshot_global_params()
        super().distribute_to_clients(client_models)


class ScaffoldAggregator(FedAvgAggregator):
    """
    scaffold aggregator for variance reduction in federated learning.

    uses control variates to correct for client drift, achieving faster
    convergence than fedavg/fedprox with fewer communication rounds.

    simplified implementation for feasibility study.

    reference:
        karimireddy et al., "scaffold: stochastic controlled averaging
        for federated learning", icml 2020.
    """

    def __init__(
        self,
        global_model: nn.Module,
        n_clients: int = 2,
        share_discriminators: bool = False,
    ):
        super().__init__(global_model, share_discriminators)
        self.n_clients = n_clients

        # control variates (one per client + global)
        self.global_control = {
            k: torch.zeros_like(v)
            for k, v in global_model.state_dict().items()
        }
        self.client_controls = [
            {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
            for _ in range(n_clients)
        ]

    def compute_correction(self, client_idx: int) -> Dict[str, torch.Tensor]:
        """
        compute gradient correction for a client.

        args:
            client_idx: index of the client
        returns:
            correction dict (global_control - client_control)
        """
        correction = {}
        for k in self.global_control:
            correction[k] = self.global_control[k] - self.client_controls[client_idx]
        return correction

    def update_controls(
        self,
        client_idx: int,
        client_model: nn.Module,
        learning_rate: float,
        local_steps: int,
    ) -> None:
        """
        update client and global control variates after local training.

        args:
            client_idx: index of the client
            client_model: client model after local training
            learning_rate: local learning rate
            local_steps: number of local gradient steps
        """
        global_state = self.global_model.state_dict()
        client_state = client_model.state_dict()

        for k in self.global_control:
            # new client control = client_control - global_control
            #   + (global_model - client_model) / (local_steps * lr)
            delta = (global_state[k].float() - client_state[k].float()) / (
                local_steps * learning_rate
            )
            new_ci = (
                self.client_controls[client_idx][k].float()
                - self.global_control[k].float()
                + delta
            )

            # update global control
            old_ci = self.client_controls[client_idx][k].float()
            self.global_control[k] = (
                self.global_control[k].float()
                + (new_ci - old_ci) / self.n_clients
            ).to(self.global_control[k].dtype)

            # update client control
            self.client_controls[client_idx][k] = new_ci.to(
                self.client_controls[client_idx][k].dtype
            )
