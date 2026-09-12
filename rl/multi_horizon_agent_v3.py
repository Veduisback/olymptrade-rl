from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from rl.config import DEVICE, RANDOM_SEED
from rl.state import state_size


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONS = (
    5.0,
    15.0,
    30.0,
    45.0,
    60.0,
    120.0,
)

NUM_HORIZONS = len(HORIZONS)
ACTIONS_PER_HEAD = 3  # WAIT, BUY, SELL


# ============================================================
# ACTION HELPERS
# ============================================================

def action_name(head: int, action: int) -> str:
    """
    Convert a horizon/action pair into a readable name.
    """

    if head < 0 or head >= NUM_HORIZONS:
        raise ValueError(
            f"Invalid horizon head: {head}"
        )

    if action == 0:
        return "WAIT"

    if action == 1:
        return f"BUY_{int(HORIZONS[head])}s"

    if action == 2:
        return f"SELL_{int(HORIZONS[head])}s"

    raise ValueError(
        f"Invalid action: {action}"
    )


# ============================================================
# HORIZON HEAD
# ============================================================

class HorizonHead(nn.Module):
    """
    Independent WAIT/BUY/SELL prediction head for one horizon.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),

            nn.Linear(64, ACTIONS_PER_HEAD),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


# ============================================================
# MULTI-HORIZON NETWORK
# ============================================================

class MultiHorizonQNetworkV3(nn.Module):
    """
    Shared local-state encoder followed by six independent
    horizon-specific WAIT/BUY/SELL heads.

    Input:
        [batch, 22]

    Output:
        [batch, 6, 3]

    Meaning:

        output[:, 0, :] -> 5s
        output[:, 1, :] -> 15s
        output[:, 2, :] -> 30s
        output[:, 3, :] -> 45s
        output[:, 4, :] -> 60s
        output[:, 5, :] -> 120s

    Each head contains:

        [WAIT, BUY, SELL]
    """

    def __init__(
        self,
        state_dim: int = 22,
    ):
        super().__init__()

        self.state_dim = int(state_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList(
            [
                HorizonHead(64)
                for _ in range(NUM_HORIZONS)
            ]
        )

    def forward(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:

        z = self.encoder(states)

        outputs = [
            head(z)
            for head in self.heads
        ]

        return torch.stack(
            outputs,
            dim=1,
        )

        # Result:
        # [batch, 6, 3]


# ============================================================
# DECISION RESULT
# ============================================================

@dataclass
class V3Decision:
    horizon: float
    action: int
    q_value: float
    q_values: np.ndarray


# ============================================================
# AGENT
# ============================================================

class MultiHorizonDQNAgentV3:

    def __init__(
        self,
        state_dim: int = 22,
        learning_rate: float = 3e-4,
        gamma: float = 0.95,
    ):
        torch.manual_seed(
            RANDOM_SEED
        )

        np.random.seed(
            RANDOM_SEED
        )

        self.device = torch.device(
            DEVICE
        )

        self.state_dim = int(
            state_dim
        )

        self.gamma = float(
            gamma
        )

        # --------------------------------------------------------
        # ONLINE NETWORK
        # --------------------------------------------------------

        self.online = (
            MultiHorizonQNetworkV3(
                state_dim=self.state_dim
            )
            .to(self.device)
        )

        # --------------------------------------------------------
        # TARGET NETWORK
        #
        # Kept for checkpoint compatibility.
        #
        # V3 direct offline learning does NOT bootstrap from it.
        # --------------------------------------------------------

        self.target = (
            MultiHorizonQNetworkV3(
                state_dim=self.state_dim
            )
            .to(self.device)
        )

        self.target.load_state_dict(
            self.online.state_dict()
        )

        self.target.eval()

        # --------------------------------------------------------
        # OPTIMIZER
        # --------------------------------------------------------

        self.optimizer = Adam(
            self.online.parameters(),
            lr=learning_rate,
        )

        self.training_steps = 0


    # ============================================================
    # Q-VALUE INFERENCE
    # ============================================================

    @torch.no_grad()
    def q_values(
        self,
        states: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Return model predictions.

        Single state:
            input  [22]
            output [6, 3]

        Batch:
            input  [B, 22]
            output [B, 6, 3]
        """

        if isinstance(
            states,
            np.ndarray,
        ):
            x = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            )

        elif isinstance(
            states,
            torch.Tensor,
        ):
            x = states.to(
                self.device,
                dtype=torch.float32,
            )

        else:
            raise TypeError(
                "states must be numpy.ndarray "
                "or torch.Tensor"
            )

        single = (
            x.ndim == 1
        )

        if single:
            if x.shape[0] != self.state_dim:
                raise ValueError(
                    f"Expected state dimension "
                    f"{self.state_dim}, "
                    f"got {x.shape[0]}"
                )

            x = x.unsqueeze(0)

        elif x.ndim == 2:
            if x.shape[1] != self.state_dim:
                raise ValueError(
                    f"Expected state dimension "
                    f"{self.state_dim}, "
                    f"got {x.shape[1]}"
                )

        else:
            raise ValueError(
                f"Expected states with 1 or 2 dimensions, "
                f"got shape {tuple(x.shape)}"
            )

        q = self.online(
            x
        ).cpu().numpy()

        if single:
            return q[0]

        return q


    # ============================================================
    # BEST DECISION
    # ============================================================

    @torch.no_grad()
    def best_decision(
        self,
        state: np.ndarray,
    ) -> V3Decision:
        """
        Select the highest predicted reward among:

            WAIT
            BUY_5
            SELL_5
            BUY_15
            SELL_15
            ...
            BUY_120
            SELL_120
        """

        q = self.q_values(
            state
        )

        # q shape:
        #
        # [6, 3]

        if q.shape != (
            NUM_HORIZONS,
            ACTIONS_PER_HEAD,
        ):
            raise RuntimeError(
                "Unexpected Q-value shape: "
                f"{q.shape}"
            )

        flat_index = int(
            np.argmax(q)
        )

        horizon_index, action = (
            np.unravel_index(
                flat_index,
                q.shape,
            )
        )

        return V3Decision(
            horizon=HORIZONS[
                horizon_index
            ],

            action=int(
                action
            ),

            q_value=float(
                q[
                    horizon_index,
                    action,
                ]
            ),

            q_values=q[
                horizon_index
            ].copy(),
        )


    # ============================================================
    # TRAIN ONE BATCH
    # ============================================================

    def train_batch(
        self,
        states: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        next_states: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
        weights: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        Train on one batch of offline samples.

        Expected shapes:

            states:
                [B, 22]

            rewards:
                [B, 6, 3]

            next_states:
                [B, 6, 22]

            dones:
                [B, 6]

        IMPORTANT:

        V3 is direct offline reward learning.

        The realized historical reward is the target.

        Therefore:

            target = reward

        We do NOT bootstrap from next_states.

        This is intentional.

        The next_states and dones arguments remain in the
        interface for compatibility with the dataset/trainer
        and for possible future RL versions.
        """

        # --------------------------------------------------------
        # CONVERT INPUTS
        # --------------------------------------------------------

        s = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        r = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        # next_states and dones are intentionally not used
        # for the V3 direct-reward objective.
        #
        # Validate them so shape mistakes are caught immediately.

        ns = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        d = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # --------------------------------------------------------
        # VALIDATE SHAPES
        # --------------------------------------------------------

        if s.ndim != 2:
            raise ValueError(
                "states must have shape [B, state_dim], "
                f"got {tuple(s.shape)}"
            )

        batch_size = s.shape[0]

        if s.shape[1] != self.state_dim:
            raise ValueError(
                f"states expected dimension "
                f"{self.state_dim}, "
                f"got {s.shape[1]}"
            )

        expected_reward_shape = (
            batch_size,
            NUM_HORIZONS,
            ACTIONS_PER_HEAD,
        )

        if tuple(r.shape) != expected_reward_shape:
            raise ValueError(
                "rewards must have shape "
                f"{expected_reward_shape}, "
                f"got {tuple(r.shape)}"
            )

        expected_next_shape = (
            batch_size,
            NUM_HORIZONS,
            self.state_dim,
        )

        if tuple(ns.shape) != expected_next_shape:
            raise ValueError(
                "next_states must have shape "
                f"{expected_next_shape}, "
                f"got {tuple(ns.shape)}"
            )

        expected_done_shape = (
            batch_size,
            NUM_HORIZONS,
        )

        if tuple(d.shape) != expected_done_shape:
            raise ValueError(
                "dones must have shape "
                f"{expected_done_shape}, "
                f"got {tuple(d.shape)}"
            )

        # --------------------------------------------------------
        # SAMPLE WEIGHTS
        # --------------------------------------------------------

        if weights is None:

            w = torch.ones(
                batch_size,
                dtype=torch.float32,
                device=self.device,
            )

        else:

            w = torch.as_tensor(
                weights,
                dtype=torch.float32,
                device=self.device,
            )

            if w.ndim != 1:
                raise ValueError(
                    "weights must have shape [B], "
                    f"got {tuple(w.shape)}"
                )

            if len(w) != batch_size:
                raise ValueError(
                    "weights length must match batch size"
                )

            w = torch.clamp(
                w,
                min=0.0,
            )

        # --------------------------------------------------------
        # CURRENT PREDICTIONS
        # --------------------------------------------------------

        q = self.online(
            s
        )

        # q:
        #
        # [B, 6, 3]

        # --------------------------------------------------------
        # DIRECT OFFLINE TARGET
        # --------------------------------------------------------

        targets = r

        # --------------------------------------------------------
        # HUBER LOSS
        # --------------------------------------------------------

        element_loss = (
            torch.nn.functional.smooth_l1_loss(
                q,
                targets,
                reduction="none",
            )
        )

        # Average the three actions and six horizons.
        #
        # [B, 6, 3]
        #       ↓
        # [B]

        per_sample_loss = (
            element_loss.mean(
                dim=(1, 2)
            )
        )

        # Apply optional sample weights.

        weighted_loss = (
            per_sample_loss * w
        )

        weight_sum = w.sum()

        if float(
            weight_sum.detach().cpu()
        ) > 0.0:

            loss = (
                weighted_loss.sum()
                / weight_sum
            )

        else:

            loss = weighted_loss.mean()

        # --------------------------------------------------------
        # OPTIMIZATION
        # --------------------------------------------------------

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            max_norm=5.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        # --------------------------------------------------------
        # PREDICTION ERRORS
        # --------------------------------------------------------

        errors = (
            (targets - q)
            .abs()
            .mean(
                dim=(1, 2)
            )
            .detach()
            .cpu()
            .numpy()
        )

        return (
            float(
                loss.detach().cpu().item()
            ),
            errors,
        )


    # ============================================================
    # TARGET UPDATE
    # ============================================================

    def update_target(
        self,
    ) -> None:
        """
        Synchronize the compatibility target network.

        V3 direct reward learning does not currently use the
        target network during optimization.
        """

        self.target.load_state_dict(
            self.online.state_dict()
        )


    # ============================================================
    # SAVE
    # ============================================================

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "version": "v3_direct_offline",
                "state_dim": self.state_dim,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "gamma": self.gamma,
                "training_steps": self.training_steps,
                "horizons": HORIZONS,
            },
            path,
        )


    # ============================================================
    # LOAD
    # ============================================================

    def load(
        self,
        path: str | Path,
    ) -> None:

        path = Path(
            path
        )

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        saved_state_dim = int(
            checkpoint.get(
                "state_dim",
                self.state_dim,
            )
        )

        if saved_state_dim != self.state_dim:
            raise ValueError(
                "Model state dimension mismatch: "
                f"checkpoint={saved_state_dim}, "
                f"current={self.state_dim}"
            )

        saved_horizons = tuple(
            float(x)
            for x in checkpoint.get(
                "horizons",
                HORIZONS,
            )
        )

        if saved_horizons != HORIZONS:
            raise ValueError(
                "Model horizon configuration mismatch: "
                f"checkpoint={saved_horizons}, "
                f"current={HORIZONS}"
            )

        self.online.load_state_dict(
            checkpoint["online"]
        )

        if "target" in checkpoint:
            self.target.load_state_dict(
                checkpoint["target"]
            )
        else:
            self.target.load_state_dict(
                self.online.state_dict()
            )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        self.gamma = float(
            checkpoint.get(
                "gamma",
                self.gamma,
            )
        )

        self.training_steps = int(
            checkpoint.get(
                "training_steps",
                0,
            )
        )


# ============================================================
# STATE DIMENSION
# ============================================================

STATE_DIM = state_size()


if STATE_DIM != 22:
    raise RuntimeError(
        f"Expected 22-dimensional state, "
        f"but state_size() returned {STATE_DIM}"
    )