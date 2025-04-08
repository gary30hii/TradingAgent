import numpy as np
import pandas as pd
from finta import TA
from enum import Enum
import matplotlib.pyplot as plt
import gymnasium as gym
from time import time


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class MultiAssetTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 3}

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2
        assert df.ndim == 2

        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.render_mode = render_mode

        self.trade_fee_bid_percent = 0.01
        self.trade_fee_ask_percent = 0.005
        self.trade_penalty = 0.01  # discourage excessive trades

        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # Gym spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32
        )

        # Episode state
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(
            int(self.np_random.uniform(0, seed if seed is not None else 1))
        )

        self._truncated = False

        # 更安全地计算 tick 范围
        max_start_tick = self.frame_bound[1] - self.frame_bound[0] - self.window_size
        self._start_tick = self.window_size + np.random.randint(0, max_start_tick)
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1

        self._position = None
        self._position_history = [self._position] * self.window_size
        self._total_reward = 0.0
        self._total_profit = 1.0
        self._first_rendering = True
        self.history = {}
        self._end_tick = (
            len(self.prices) - 1
        )  # 使用实际 prices 长度，避免 frame_bound 越界

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._truncated = False
        self._current_tick += 1

        # 越界保护：提前终止
        if self._current_tick >= len(self.prices):
            self._truncated = True
            return (
                self._get_observation(),
                0.0,
                False,
                self._truncated,
                self._get_info(),
            )

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        if action == Actions.Buy.value and self._position != Positions.Long:
            self._position = Positions.Long
            self._last_trade_tick = self._current_tick

        elif action == Actions.Sell.value and self._position != Positions.Short:
            self._position = Positions.Short
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        info["action"] = action
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        terminated = self._total_profit < 0.1
        return observation, step_reward, terminated, self._truncated, info

    def _process_data(self):
        df = self.df.copy()

        # Technical indicators
        df["SMA"] = TA.SMA(df, 12)
        df["MAI_1,20"] = TA.SMA(df, 1) / (TA.SMA(df, 20) + 1e-6)
        df["MAI_10,40"] = TA.SMA(df, 10) / (TA.SMA(df, 40) + 1e-6)
        df["ROC_6"] = TA.ROC(df, 6)
        df["ROC_15"] = TA.ROC(df, 15)
        df["RSI_6"] = TA.RSI(df, 6) / 100.0
        df["RSI_12"] = TA.RSI(df, 12) / 100.0
        df["K_9/D_9"] = TA.STOCH(df, 9) / (TA.STOCHD(df, 9) + 1e-6)
        df["K_15/D_15"] = TA.STOCH(df, 15) / (TA.STOCHD(df, 15) + 1e-6)
        df["K_15"] = TA.STOCH(df, 15) / 100.0
        df["D_15"] = TA.STOCHD(df, 15) / 100.0
        df["OBV"] = TA.OBV(df)
        df["OBV"] = (df["OBV"] - df["OBV"].mean()) / (df["OBV"].std() + 1e-6)

        df["Price_Change_Rate"] = df["close"].pct_change().fillna(0)
        df["Volume_Change_Rate"] = df["volume"].pct_change().fillna(0)

        df.fillna(0, inplace=True)

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        # 限定范围后再取出 signal_features 和 close
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        # 裁剪用于 simulation 的数据
        df_subset = df.iloc[start:end].copy()

        prices = df_subset["close"].to_numpy()
        signal_features = df_subset[
            [
                "SMA",
                "MAI_1,20",
                "MAI_10,40",
                "ROC_6",
                "ROC_15",
                "RSI_6",
                "RSI_12",
                "K_9/D_9",
                "K_15/D_15",
                "K_15",
                "D_15",
                "OBV",
                "Price_Change_Rate",
                "Volume_Change_Rate",
            ]
        ].to_numpy()

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        if self._position is None:
            return 0.0

        current_price = self._safe_price(self.prices[self._current_tick])
        last_trade_price = self._safe_price(self.prices[self._last_trade_tick])

        virtual_profit = self._total_profit  # 当前总资产（作为起点）

        if self._position == Positions.Long:
            shares = (
                virtual_profit * (1 - self.trade_fee_ask_percent) / last_trade_price
            )
            virtual_profit = shares * (1 - self.trade_fee_bid_percent) * current_price

        elif self._position == Positions.Short:
            shares = (
                virtual_profit * (1 - self.trade_fee_bid_percent) / last_trade_price
            )
            gain = max(2 * last_trade_price - current_price, 1e-6)
            virtual_profit = shares * (1 - self.trade_fee_ask_percent) * gain

        # 避免 reward 爆炸或负资产
        virtual_profit = max(virtual_profit, 1e-6)

        reward = (virtual_profit - self._total_profit) / self._total_profit

        is_trade = (
            action == Actions.Buy.value and self._position != Positions.Long
        ) or (action == Actions.Sell.value and self._position != Positions.Short)

        if is_trade:
            reward -= self.trade_penalty

        return reward

    def _update_profit(self, action):
        if self._position is None:
            return 0.0

        trade = (action == Actions.Buy.value and self._position != Positions.Long) or (
            action == Actions.Sell.value and self._position != Positions.Short
        )

        if trade or self._truncated:
            current_price = self._safe_price(self.prices[self._current_tick])
            last_trade_price = self._safe_price(self.prices[self._last_trade_tick])

            if self._position == Positions.Long:
                shares = (
                    self._total_profit
                    * (1 - self.trade_fee_ask_percent)
                    / last_trade_price
                )
                self._total_profit = (
                    shares * (1 - self.trade_fee_bid_percent) * current_price
                )

            elif self._position == Positions.Short:
                shares = (
                    self._total_profit
                    * (1 - self.trade_fee_bid_percent)
                    / last_trade_price
                )
                gain = max(2 * last_trade_price - current_price, 1e-6)
                self._total_profit = shares * (1 - self.trade_fee_ask_percent) * gain

        self._total_profit = max(self._total_profit, 1e-6)

    def _get_observation(self):
        return self.signal_features[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]

    def _get_info(self):
        return {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "position": self._position,
            "tick": self._current_tick,
        }

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info}
        for key, value in info.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def render(self, mode="human"):
        def _plot_position(position, tick):
            color = "green" if position == Positions.Long else "red"
            plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            _plot_position(self._position, self._current_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            f"Total Reward: {self._total_reward:.6f} ~ Total Profit: {self._total_profit:.6f}"
        )

        process_time = time() - start_time
        pause_time = (1 / self.metadata["render_fps"]) - process_time
        if pause_time > 0:
            plt.pause(pause_time)

    def render_all(self, title=None):
        ticks = np.arange(len(self._position_history))
        if len(ticks) > len(self.prices):
            ticks = ticks[: len(self.prices)]

        short_ticks = [
            i
            for i in ticks
            if self._position_history[i] is not None
            and self._position_history[i] == Positions.Short
        ]

        long_ticks = [
            i
            for i in ticks
            if self._position_history[i] is not None
            and self._position_history[i] == Positions.Long
        ]

        none_ticks = [i for i in ticks if self._position_history[i] is None]

        plt.figure(figsize=(15, 5))
        plt.plot(self.prices, label="Price", color="blue")
        plt.plot(
            short_ticks, [self.prices[i] for i in short_ticks], "ro", label="Short"
        )
        plt.plot(long_ticks, [self.prices[i] for i in long_ticks], "go", label="Long")
        plt.plot(
            none_ticks,
            [self.prices[i] for i in none_ticks],
            "ko",
            alpha=0.3,
            label="No Position",
        )

        if title:
            plt.title(title)

        plt.suptitle(
            f"Total Reward: {self._total_reward:.6f} ~ Total Profit: {self._total_profit:.6f}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0  # 初始资产

        while current_tick <= self._end_tick:
            # 根据趋势判断建仓方向
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                # 下跌趋势 → 做空
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] < self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Short
            else:
                # 上涨趋势 → 做多
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] >= self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Long

            exit_price = self._safe_price(self.prices[current_tick - 1])
            entry_price = self._safe_price(self.prices[last_trade_tick])

            if position == Positions.Long:
                # 买入时付 ask 手续费，卖出时付 bid 手续费
                shares = profit * (1 - self.trade_fee_ask_percent) / entry_price
                profit = shares * (1 - self.trade_fee_bid_percent) * exit_price

            elif position == Positions.Short:
                # 卖出（借股）时付 bid 手续费，回补买入时付 ask 手续费
                shares = profit * (1 - self.trade_fee_bid_percent) / entry_price
                profit = (
                    shares
                    * (1 - self.trade_fee_ask_percent)
                    * (2 * entry_price - exit_price)
                )

            last_trade_tick = current_tick - 1

        return profit

    def _safe_price(self, price):
        return np.clip(price, 1e-6, 1e6)  # 限制价格在合理范围

    def visualize_history(df):
        plt.figure(figsize=(14, 6))

        # 绘制总资产曲线
        plt.plot(df["tick"], df["total_profit"], label="Total Profit", color="blue")
        df["action_name"] = df["action"].map({0: "Hold", 1: "Buy", 2: "Sell"})
        print(df[["tick", "action", "action_name"]].tail())

        # 标出买入（action=1）和卖出（action=2）
        buy_ticks = df[df["action"] == Actions.Buy.value]["tick"]
        sell_ticks = df[df["action"] == Actions.Sell.value]["tick"]
        buy_profits = df[df["action"] == Actions.Buy.value]["total_profit"]
        sell_profits = df[df["action"] == Actions.Sell.value]["total_profit"]

        plt.scatter(
            buy_ticks, buy_profits, marker="^", color="green", label="Buy", zorder=5
        )
        plt.scatter(
            sell_ticks, sell_profits, marker="v", color="red", label="Sell", zorder=5
        )

        plt.title("Total Profit Over Time with Buy/Sell Points")
        plt.xlabel("Tick")
        plt.ylabel("Profit")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def export_history(self):
        return pd.DataFrame(self.history)
