# ASIA_X Multi-Horizon RL Research

AI-based research project for analyzing **ASIA_X (Asia Composite Index)** quote data using a multi-horizon neural network.

The project explores whether a learned model can estimate the directional outcome of different future time horizons from historical price/quote sequences.

> **Research / educational project.**
> This repository is not intended to provide financial advice or guarantee profitable trading decisions.

---

## Overview

The project uses historical ASIA_X quote data to train a neural network that evaluates multiple possible actions across several future horizons.

The model considers:

* `WAIT`
* `BUY`
* `SELL`

across six prediction horizons:

```text
5s
15s
30s
45s
60s
120s
```

Instead of using manually written trading rules, the model learns relationships between the input state and historical outcomes.

---

## Architecture

The current model is a multi-head neural network.

```text
Quote History
     │
     ▼
State Representation
     │
     ▼
128-unit layer
     │
     ▼
128-unit layer
     │
     ▼
64-unit layer
     │
     ├────────► 5s   → WAIT / BUY / SELL
     ├────────► 15s  → WAIT / BUY / SELL
     ├────────► 30s  → WAIT / BUY / SELL
     ├────────► 45s  → WAIT / BUY / SELL
     ├────────► 60s  → WAIT / BUY / SELL
     └────────► 120s → WAIT / BUY / SELL
```

### Model configuration

| Property            |                      Value |
| ------------------- | -------------------------: |
| State size          |                         22 |
| Hidden layers       |             128 → 128 → 64 |
| Horizons            | 5, 15, 30, 45, 60, 120 sec |
| Actions per horizon |                          3 |
| Total outputs       |                         18 |
| Discount factor     |                       0.95 |
| Learning rate       |                       3e-4 |
| Framework           |                    PyTorch |

---

## Important Training Note

Although the project uses the name **DQN**, the current V3 implementation is more accurately described as **direct offline reward prediction**.

The training target is the historical reward for each action/horizon combination.

The current training step effectively learns:

```text
state → expected historical reward
```

rather than performing a traditional bootstrapped DQN update using:

```text
reward + gamma × max(next_state_Q)
```

This distinction is intentional and documented as part of the current research stage.

---

## Dataset

Raw quote files use the following format:

```csv
quote_index,timestamp,price,pair
0,1720000000.123,6153.12,ASIA_X
1,1720000000.641,6153.18,ASIA_X
2,1720000001.205,6153.14,ASIA_X
```

### Dataset structure

The data is divided into numbered segments.

Training:

```text
segments 0–20
```

Unseen evaluation:

```text
segments 21–25
```

The unseen segments are kept separate from training so that model performance can be evaluated on data that was not used during training.

---

## State Representation

The model uses a state vector of size:

```text
22
```

The state is constructed from recent quote history and derived market information.

A lookback window is used to provide the model with short-term temporal context rather than a single isolated price.

---

## Reward System

For each horizon, the dataset generates rewards for all three actions.

Conceptually:

```text
WAIT → wait reward

BUY  → positive reward if future price > entry price
       negative reward otherwise

SELL → positive reward if future price < entry price
       negative reward otherwise
```

Current reward values include:

```text
WIN_REWARD  = +0.85
LOSS_REWARD = -1.0
```

The reward structure reflects an 85% payout assumption used during the research experiments.

---

## Training Pipeline

The general workflow is:

```text
Raw Quote CSV
      │
      ▼
Dataset Loader
      │
      ▼
State Construction
      │
      ▼
Multi-Horizon Reward Generation
      │
      ▼
Neural Network Training
      │
      ▼
Model Checkpoint
      │
      ▼
Unseen Evaluation
```

Training uses segments `0–20`.

Evaluation uses separate segments `21–25`.

---

## Repository Structure

```text
olymptrade-rl/
│
├── rl/
│   ├── multi_horizon_agent_v3.py
│   ├── multi_horizon_dataset_v3.py
│   ├── train_multi_horizon_v3.py
│   ├── live_ai_trader_v1.py
│   └── ...
│
├── models/
│   └── rl/
│       └── multi_horizon_dqn_v3.pt
│
├── data/
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
```

Large datasets, trained checkpoints, logs and other generated files are intentionally excluded from Git where appropriate.

---

## Installation

Clone the repository:

```powershell
git clone https://github.com/Veduisback/olymptrade-rl.git
cd olymptrade-rl
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Training

The training script can be run as a Python module:

```powershell
python -m rl.train_multi_horizon_v3
```

Training uses the configured training segments and saves the resulting model checkpoint.

---

## Evaluation

The model should be evaluated against data that was not used during training.

The current evaluation set is:

```text
segments 21–25
```

This separation is important because evaluating on the training data can give an overly optimistic estimate of model performance.

---

## Current Evaluation

An evaluation on unseen segments produced approximately:

| Horizon | BUY Accuracy |
| ------: | -----------: |
