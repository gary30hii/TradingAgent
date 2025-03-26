import gym
from gym import spaces
import numpy as np
import pandas as pd


class SingleEquityTradingEnv(gym.Env):
    """A simple custom environment for trading a single stock using OHLCV data."""

    def __init__(self, df, initial_cash=10_000):
        super(SingleEquityTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.max_steps = len(df) - 1

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [Price, Cash, Shares Held]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.last_net_worth = self.initial_cash
        return self._get_observation()

    def _get_observation(self):
        """Returns the current state: [price, cash, shares held]"""
        price = self.df.loc[self.current_step, "Close"]
        return np.array([price, self.cash, self.shares_held], dtype=np.float32)

    def step(self, action):
        """Executes the action and returns (obs, reward, done, info)"""
        done = False
        price = self.df.loc[self.current_step, "Close"]

        # Execute action
        if action == 1:  # Buy
            if self.cash >= price:
                self.shares_held += 1
                self.cash -= price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.cash += price

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        new_net_worth = (
            self.cash + self.shares_held * self.df.loc[self.current_step, "Close"]
        )
        reward = new_net_worth - self.last_net_worth
        self.last_net_worth = new_net_worth

        obs = self._get_observation()
        return obs, reward, done, {}

    def render(self, mode="human"):
        """Prints current step info."""
        price = self.df.loc[self.current_step, "Close"]
        net_worth = self.cash + self.shares_held * price
        print(
            f"Step: {self.current_step} | Price: {price:.2f} | Cash: {self.cash:.2f} | Shares: {self.shares_held} | Net Worth: {net_worth:.2f}"
        )
