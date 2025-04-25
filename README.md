# 📈 TradingAgent

**TradingAgent** is a coursework project focused on comparing **rule-based algorithms** and **Deep Reinforcement Learning (DRL)-based algorithms** for portfolio management. The project further investigates the impact of including cryptocurrencies on overall portfolio performance.

This work utilizes the **FinRL framework** developed by the AI4Finance Foundation to implement and simulate DRL-based trading strategies.

> **Framework Reference:**  
> AI4Finance Foundation, FinRL: Deep Reinforcement Learning Framework for Automated Trading. [GitHub Repository](https://github.com/AI4Finance-Foundation/FinRL)

---

## 🎯 Project Objective

- Compare **rule-based methods** (e.g., Minimum Variance Optimization, Naive Diversification, Mean-Variance Optimization, Rolling Mean-Variance Optimization) with **DRL-based methods** (A2C, PPO, DDPG, SAC, TD3, and Ensemble strategies) in managing multi-asset portfolios.
- Evaluate how the **inclusion of cryptocurrencies** affects portfolio performance across different strategies.
- Use real-world financial data spanning stocks and crypto assets to simulate realistic trading environments.

---

## 📂 Project Structure

### Notebooks Overview

- **`data.ipynb`**  
  Data acquisition for both stock and cryptocurrency markets.

- **`data_prep.ipynb`**  
  Data preprocessing and feature engineering for model readiness.

- **DRL-Based Strategies:**  
  `PortfolioAllocationModelTraining(a2c&ddpg&ppo&sac&td3).ipynb`  
  `PortfolioAllocationModelTraining(ensemble).ipynb`  

- **Rule-Based Strategies:**  
  `PortfolioAllocationModelTraining(min_variance_optimization).ipynb`  
  `PortfolioAllocationModelTraining(traditional_adaptive_mean_variance_optimization).ipynb`  

- **Evaluation:**  
  `evaluation.ipynb` — Performance comparison across all strategies.

---

## 📊 Datasets

Prepared datasets used in experiments:

- `2007-2025_no_crypto.csv`  *(Stock-only portfolio)*
- `2015-2025_crypto.csv`      *(Crypto-focused portfolio)*
- `2015-2025_no_crypto.csv`  *(Recent stock-only portfolio)*

Each experiment's output is saved in directories named after the dataset file.


---