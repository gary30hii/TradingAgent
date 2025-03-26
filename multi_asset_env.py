import numpy as np
import pandas as pd
from finta import TA
from trading_env import TradingEnv, Actions, Positions


class MultiAssetTradingEnv(TradingEnv):
    def __init__(
        self,
        df,
        window_size,
        frame_bound,
        render_mode=None,
        trade_fee_bid_percent=0.01,
        trade_fee_ask_percent=0.005,
        asset_name="GenericAsset",
    ):
        self.frame_bound = frame_bound
        self.trade_fee_bid_percent = trade_fee_bid_percent
        self.trade_fee_ask_percent = trade_fee_ask_percent
        self.asset_name = asset_name
        self.returns_history = []

        super().__init__(df, window_size, render_mode)

    def step(self, action):
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        ):
            trade = True

        # Flip position BEFORE updating profit to use the correct previous position
        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        # Optional debug print
        print(
            f"Tick: {self._current_tick}, Action: {Actions(action).name}, Position: {self._position.name}, Price: {self.prices[self._current_tick]}"
        )

        return observation, step_reward, False, self._truncated, info

    def _process_data(self):
        df = self.df.copy()

        # --- Indicator Calculations using finta ---
        df["SMA"] = TA.SMA(df, 12)
        df["RSI"] = TA.RSI(df)
        df["OBV"] = TA.OBV(df)

        df.fillna(0, inplace=True)

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        prices = df.iloc[start:end]["close"].to_numpy()
        signal_features = df.iloc[start:end][["SMA", "RSI", "OBV"]].to_numpy()

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        # Get prices
        current_price = self.prices[self._current_tick]
        prev_price = self.prices[self._current_tick - 1]

        # Calculate return z_t
        z_t = (current_price - prev_price) / prev_price
        self.returns_history.append(z_t)

        # Infer current position (e_t) after action
        if self._position == Positions.Short and action == Actions.Buy.value:
            e_t = 1  # switch to long
        elif self._position == Positions.Long and action == Actions.Sell.value:
            e_t = -1  # switch to short
        elif self._position == Positions.Long:
            e_t = 1  # hold long
        elif self._position == Positions.Short:
            e_t = -1  # hold short
        else:
            e_t = 0  # neutral

        # Infer previous position (e_prev)
        if len(self._position_history) > 0:
            prev_position = self._position_history[-1]
            if prev_position == Positions.Long:
                e_prev = 1
            elif prev_position == Positions.Short:
                e_prev = -1
            else:
                e_prev = 0
        else:
            e_prev = e_t  # fallback on first tick

        # PnL reward
        r_pnl = e_t * z_t

        # Commission penalty
        commission = 0.001
        r_com = -commission * abs(e_t - e_prev)

        # Sharpe Ratio reward
        r_sharpe = 0
        alpha = 1.0
        w = len(self.returns_history) // 2  # dynamic window

        if len(self.returns_history) >= w and w > 1:
            returns_window = self.returns_history[-w:]
            mean_return = np.mean(returns_window)
            std_return = np.std(returns_window)

            if std_return != 0:
                r_sharpe = alpha * (mean_return / std_return)

                # Previous Sharpe to apply penalty if needed
                if len(self.returns_history) > w:
                    prev_window = self.returns_history[-(w + 1) : -1]
                    prev_sharpe = (
                        np.mean(prev_window) / np.std(prev_window)
                        if np.std(prev_window) != 0
                        else 0
                    )
                    if r_sharpe < prev_sharpe:
                        r_sharpe = -r_sharpe  # apply penalty

        total_reward = r_pnl + r_com + r_sharpe
        return total_reward

    def _update_profit(self, action):
        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                # Close Long: sell at current_price
                shares = (
                    self._total_profit * (1 - self.trade_fee_ask_percent)
                ) / last_trade_price
                self._total_profit = (
                    shares * (1 - self.trade_fee_bid_percent)
                ) * current_price

            elif self._position == Positions.Short:
                # Close Short: buy back at current_price
                shares = (
                    self._total_profit * (1 - self.trade_fee_bid_percent)
                ) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_ask_percent)) * (
                    2 * last_trade_price - current_price
                )

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] < self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Short
            else:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] >= self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.returns_history = []  # clear return history every episode
        return observation, info
