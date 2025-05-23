# CS6700 Programming Assignment 3: Hierarchical Reinforcement Learning (Taxi Domain)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](requirements.txt)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-brightgreen.svg)](https://pypi.org/project/gymnasium/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-orange.svg)](https://pypi.org/project/numpy/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.5+-blueviolet.svg)](https://pypi.org/project/matplotlib/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-yellow.svg)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13+-red.svg)](https://pytorch.org/)
[![tqdm](https://img.shields.io/badge/tqdm-progress--bar-lightgrey.svg)](https://pypi.org/project/tqdm/)


An implementation and comparison of **SMDP Q-Learning** and **Intra-Option Q-Learning** for solving the **Taxi Domain** problem using **OpenAI Gymnasium**.
---

## 🧪 Objectives

1. **Implement SMDP Q-Learning:** Single-step Semi-Markov Q-Learning with pre-defined options.
2. **Implement Intra-Option Q-Learning:** Simultaneous learning of option values and intra-option policies.
3. **Analyze & Compare:**

   * Reward curves per episode
   * Learned Q-value heatmaps
   * Qualitative descriptions of option behaviors
4. **Experiment:** Evaluate an alternate, mutually exclusive option set and compare performance.

---

## 📋 Table of Contents

1. [Overview](#📝-overview)
2. [Environment: Taxi Domain](#🚕-environment-taxi-domain)
3. [Algorithms & Options](#⚙️-algorithms--options)
4. [Hyperparameters](#🎛️-hyperparameters)
5. [Project Structure](#🗂️-project-structure)
6. [Setup & Installation](#🚀-setup--installation)
7. [Usage](#▶️-usage)
8. [Results](#📊-results)
9. [References](#🔖-references)


---

## 📝 Overview

This assignment implements two hierarchical RL algorithms to solve the Taxi‑v3 environment:

* **SMDP Q-Learning:** Learns option‑value function $Q(s, o)$ with termination and initiation sets.
* **Intra-Option Q-Learning:** Learns both $Q(s, o)$ and intra-option $Q(s, a)$ concurrently, enabling improvement of option policies during training.

We compare convergence speed, sample efficiency, and the impact of option design on final performance.

---

## 🚕 Environment: Taxi Domain

* **Grid:** 5×5, four pickup/dropoff locations labeled R, G, B, Y.
* **States (500):** Taxi row, column; passenger location (four sites or in-taxi); destination.
* **Actions (6):** North, South, East, West, Pickup, Dropoff.
* **Rewards:** Step −1; successful drop‑off +20; illegal pickup/dropoff −10.
* **Discount factor (γ):** 0.9
* **Gymnasium reference:** [Taxi-v3 docs](https://gymnasium.farama.org/environments/toy_text/taxi/)

### 🖼️ Environment Map

![Taxi map](https://github.com/ahmecse/RL-Assignments-3-IITM-CS6700/raw/main/images/Visualizing%20Taxi-v3%20environment%20with%20positions%20annotated.png)

---

## ⚙️ Algorithms & Options

### Option Definitions (Initial Set)

* **Go-To-R/G/B/Y:** Moves taxi (via shortest path) to one of the four stations; terminates on arrival.
* **Primitive Options:** Pickup and Dropoff executed only if valid; terminate immediately.

### Alternate Option Set (Experiment)

* **Row-Complete:** Drive to correct row of target location, then terminate.
* **Column-Complete:** Drive to correct column of target location, then terminate.
* **Pickup/Dropoff:** As above.

### Algorithmic Details

* **SMDP Q-Learning:** Bellman update over options with duration $k$ and cumulative reward $R$.
* **Intra-Option Q-Learning:** Off-policy updates to both $Q(s,o)$ and intra-option $Q(s,a)$ per Sutton & Precup (1999).

---

## 🎛️ Hyperparameters

| Parameter         | Value              | Notes               |
| ----------------- | ------------------ | ------------------- |
| Learning rate (α) | 0.1                | Constant            |
| Discount (γ)      | 0.9                | —                   |
| ε‑greedy start    | 1.0 → 0.1 over 50k | Linear decay        |
| Episodes          | 100,000            | Converges by \~80k  |
| Batch updates     | Online             | Single-step updates |

---

## 🗂️ Project Structure

```
CS6700-PA3-HRL-Taxi/
├── docs/
│   ├── CS6700_PA3.pdf          # Assignment spec
│   └── report.pdf              # Detailed report
├── images/                     # Static visuals
├── notebooks/
│   └── hrl_taxi_domain_analysis.ipynb  # Code & experiments
├── results/
│   ├── comparison_plots/
│   │   ├── smdp_q_learning_rewards.png
│   │   ├── intra_option_q_learning_rewards.png
│   │   └── comparison_plot.png
│   └── q_values/
│       ├── smdp_q_values.png
│       └── intraop_q_values.png
├── src/
│   ├── agents/
│   ├── options/
│   └── utils/
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

```bash
git clone https://github.com/ahmecse/CS6700-PA3-HRL-Taxi.git
cd CS6700-PA3-HRL-Taxi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Usage

Run all experiments in Jupyter Lab:

```bash
cd notebooks
jupyter lab
```

Open `hrl_taxi_domain_analysis.ipynb`, set hyperparameters, and execute cells.

---

## 📊 Results

### Reward Curves: Individual & Comparison


| SMDP: Average reward per episode | IntraOP: Average reward per episode | Comparison: SMDP vs. Intra-Option |
|:--------------------------------:|:----------------------------------:|:---------------------------------:|
| ![](https://github.com/ahmecse/RL-Assignments-3-IITM-CS6700/raw/main/results/smdp/rewards.jpg) | ![](https://github.com/ahmecse/RL-Assignments-3-IITM-CS6700/raw/main/results/intraop/rewards.jpg) | ![](https://github.com/ahmecse/RL-Assignments-3-IITM-CS6700/raw/main/results/comparison_plots/comparison_plot.png) |



---

### Update Frequency Analysis
![](results/comparison_plots/Update%20Frequency_SMDP%20and%20Intra-Option%20Q-Learning.PNG)  
*How often each algorithm’s Bellman updates fire over training*

---

### Learned Q-Values
<div style="display: flex; gap: 1em;">
  <img src="https://github.com/ahmecse/RL-Assignments-3-IITM-CS6700/raw/main/results/smdp/a500_e100_g90_updates.jpg" alt="SMDP update frequency heatmap" style="width:45%;" />
  <img src="https://github.com/ahmecse/RL-Assignments-3-IITM-CS6700/raw/main/results/intraop/a500_e1_g90_updates.jpg" alt="Intra-Option update frequency heatmap" style="width:45%;" />
</div>

---

## 🔖 References

1. Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi‑MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1‑2), 181–211.
2. Gymnasium Taxi‑v3 Environment. Retrieved from [https://gymnasium.farama.org/environments/toy\_text/taxi/](https://gymnasium.farama.org/environments/toy_text/taxi/)

---

