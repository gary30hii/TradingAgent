import numpy as np
import pandas as pd
from finta import TA
from trading_env import TradingEnv, Actions, Positions


class MultiAssetTradingEnv(TradingEnv):
    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.trade_fee_bid_percent = 0.01
        self.trade_fee_ask_percent = 0.005

        super().__init__(df, window_size, render_mode)

    def _process_data(self):
        df = self.df.copy()

        # --- Indicator Calculations using finta ---
        df["SMA"] = TA.SMA(df, 12)
        df["MAI_1,20"] = TA.SMA(df, 1) / TA.SMA(df, 20)
        df["MAI_10,40"] = TA.SMA(df, 10) / TA.SMA(df, 40)
        df["ROC_6"] = TA.ROC(df, 6)
        df["ROC_15"] = TA.ROC(df, 15)
        df["RSI_6"] = TA.RSI(df, 6)
        df["RSI_12"] = TA.RSI(df, 12)
        df["K_9/D_9"] = TA.STOCH(df, 9) / TA.STOCHD(df, 9)
        df["K_15/D_15"] = TA.STOCH(df, 15) / TA.STOCHD(df, 15)
        df["K_15"] = TA.STOCH(df, 15)
        df["D_15"] = TA.STOCHD(df, 15)
        df["OBV"] = TA.OBV(df)

        # --- Additional Features ---
        df["Price_Change_Rate"] = df["close"].pct_change().fillna(0)
        df["Volume_Change_Rate"] = df["volume"].pct_change().fillna(0)

        # Fill any remaining NaNs (e.g., from indicators)
        df.fillna(0, inplace=True)

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        prices = df.iloc[start:end]["close"].to_numpy()

        # Include the new features in signal_features
        signal_features = df.iloc[start:end][
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
        step_reward = 0

        trade = (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        )

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff
            elif self._position == Positions.Short:
                step_reward += -price_diff

        return step_reward

    def _update_profit(self, action):
        trade = (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        )

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (
                    self._total_profit * (1 - self.trade_fee_ask_percent)
                ) / last_trade_price
                self._total_profit = (
                    shares * (1 - self.trade_fee_bid_percent)
                ) * current_price

            elif self._position == Positions.Short:
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
