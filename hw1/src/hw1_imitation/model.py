"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.loss_fn = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], chunk_size * action_dim),
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        # 1. Sample action
        sampled_action = self.sample_actions(state)
        # 2. Sampled action과 action_chunk의 MSE Loss 계산
        loss = self.loss_fn(sampled_action, action_chunk)
        # 3. Loss 반환
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        # 1. state로부터 action_chunk 예측
        action_chunk = self.model(state)
        # 2. 예측된 action_chunk 반환
        action_chunk = action_chunk.view(-1, self.chunk_size, self.action_dim)
        return action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.flow_matching_loss_fn = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Linear(state_dim + chunk_size * action_dim + 1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], chunk_size * action_dim),
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        # 1. 노이즈 샘플링: A_{t,0} ~ N(0,I), action_chunk과 같은 shape
        noisy_action = torch.randn_like(action_chunk)
        #
        # 2. τ 샘플링: τ ~ U(0,1), shape = (batch_size, 1)
        #    hint: torch.rand(batch_size, 1, device=...)
        tau = torch.rand(batch_size, 1, device=action_chunk.device)
        #
        # 3. action_chunk을 flatten: (batch, chunk_size * action_dim)
        #    hint: action_chunk.view(batch_size, -1)
        action_chunk_flat = action_chunk.view(batch_size, -1)
        # 4. interpolation: A_{t,τ} = τ * A_t + (1 - τ) * A_{t,0}
        interpolated_action = tau * action_chunk_flat + (1 - tau) * noisy_action.view(batch_size, -1)
        # 5. 네트워크 입력 concat: [state, A_{t,τ}, τ] → model 통과 → velocity 예측
        #    hint: torch.cat([state, interpolated_action, tau], dim=-1)
        model_input = torch.cat([state, interpolated_action, tau], dim=-1)
        # 6. 타겟 velocity 계산: A_t - A_{t,0} (원본 action - noise)
        target_velocity = action_chunk_flat - noisy_action.view(batch_size, -1)
        # 7. 예측 velocity와 타겟 velocity 사이의 MSE loss 반환
        #    hint: nn.functional.mse_loss(predicted_velocity, target_velocity)
        predicted_velocity = self.model(model_input)
        loss = self.flow_matching_loss_fn(predicted_velocity, target_velocity)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # 수식 (3) Euler integration: A_{t,τ+1/n} = A_{t,τ} + (1/n) * v_θ(o_t, A_{t,τ}, τ)
        
        # 1. 순수 노이즈에서 시작: A_{t,0} ~ N(0,I)
        #    shape = (batch_size, chunk_size * action_dim)
        batch_size = state.shape[0]
        current_action = torch.randn(batch_size, self.chunk_size * self.action_dim, device=state.device)
        # 2. num_steps번 반복 (i = 0, 1, ..., n-1):
        #    a) 현재 τ = i / num_steps, shape = (batch_size, 1)
        #    b) 네트워크 입력 concat: [state, A_{t,τ}, τ]
        #    c) velocity 예측: v = model(input)
        #    d) Euler update: A_{t,τ} = A_{t,τ} + (1 / num_steps) * v
        for i in range(num_steps):
            tau = torch.full((batch_size, 1), fill_value=i / num_steps, device=state.device)
            model_input = torch.cat([state, current_action, tau], dim=-1)
            predicted_velocity = self.model(model_input)
            current_action = current_action + (1 / num_steps) * predicted_velocity
        # 3. 최종 결과를 (batch, chunk_size, action_dim)으로 reshape하여 반환
        #    hint: result.view(-1, self.chunk_size, self.action_dim)
        return current_action.view(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
