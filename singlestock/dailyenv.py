
from typing import List
import datetime as dt
import os
from statistics import stdev

import gym
from gym import spaces
import numpy as np
import pandas as pd

import quantstats as qs

class DailyTradingEnv(gym.Env):

    def __init__(self, tickers: List, start_date: dt.datetime, end_date: dt.datetime) -> None:
        super(DailyTradingEnv, self).__init__()
        self.tickers = tickers
        self.start = start_date
        self.end_date = end_date

        self.day_data_df = {}
        self.directions_df = {}
        self.dividend_df = {}

        self.reward_range = (0, np.inf)
        # self.action_space = spaces.MultiDiscrete([5 for _ in range(len(tickers) + 1)])
        self.action_space = spaces.MultiBinary(len(tickers) + 1)
        # self.action_space = spaces.Box(low=0, high=np.inf, shape=(len(tickers) + 1,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(tickers),))
        # self.observation_space = spaces.MultiDiscrete([11 for _ in range(len(tickers))])

        for ticker in tickers:
            self.day_data_df[ticker] = pd.read_csv(os.path.join("data", "day_data", f"{ticker} MK Equity.csv"), parse_dates=True, index_col="Dates").loc[start_date:end_date]
            self.dividend_df[ticker] = pd.read_csv(os.path.join("data", "dividends", f"{ticker} dividend.csv"), parse_dates=True, index_col="Date").loc[start_date:end_date]
            self.directions_df[ticker] =  pd.read_csv(os.path.join("data", "directions", f"Directions {ticker}.csv"), parse_dates=True, index_col="Dates").loc[start_date:end_date][1:]

        self.period_length = len(self.directions_df[tickers[0]]) - 1
        assert all([(self.day_data_df[ticker].index[1:] == self.directions_df[ticker].index).all() for ticker in self.tickers])

        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0

    def step(self, action: np.ndarray):
        assert len(action) == len(self.tickers) + 1

        normalize = np.sum(action)
        if normalize > 0:
            weights = action / np.sum(action)
        else:
            weights = action
        
        curr_prices = self._get_prices(self.curr_index)
        next_prices = self._get_prices(self.curr_index + 1)
        next_day_div = self._get_div(self.curr_index)

        shares = np.floor(self.current_balance * weights[1:] / curr_prices)
        profits = np.dot(next_prices - curr_prices, shares)
        div_received = np.dot(next_day_div, shares)

        reward = self._calc_reward(profits, div_received)

        self.current_balance += profits + div_received
        self.curr_index += 1

        self.last_reward = reward

        return self._get_obs(), reward, self.curr_index == self.period_length, {}

    def reset(self):
        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0
        return self._get_obs()

    def render(self, mode="human"):
        return super().render(mode)

    def _calc_reward(self, profit, div_received):
        reward = (profit + div_received) / self.current_balance
        if reward < 0:
            reward *= 1.2
        return reward

    def _get_obs(self):
        directions = []
        for ticker in self.tickers:
            directions.append(np.sum([1 if self.directions_df[ticker][f"MODEL_{i+1}"].iloc[self.curr_index] == 1 else 0 for i in range(10)], dtype=np.float) / 10)
        return np.array(directions)

    def _get_prices(self, idx):
        prices = []
        for ticker in self.tickers:
            prices.append(self.day_data_df[ticker]["PX_LAST"].iloc[idx])
        return np.array(prices)

    def _get_div(self, idx):
        dividends = []
        date = self.directions_df[self.tickers[0]].index[idx]
        for ticker in self.tickers:
            if date in self.dividend_df[ticker].index.values:
                dividends.append(self.dividend_df[ticker]["Dividends"].loc[date])
            else:
                dividends.append(0)
        return np.array(dividends)
            

if __name__ == "__main__":
    env = DailyTradingEnv(["NESZ", "KLK", "CIMB", "MAY"], dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1))

    obs = env.reset()
    print()
    do = False
    bal = [100000]
    while not do:
        action = np.array([0] + [1 if o == 1 else 0 for o in obs])
        obs, rew, do, inf = env.step(action)
        print(obs)
        print(rew)
        print(env.current_balance)
        print(do)
        print()
        bal.append(env.current_balance)

    bal = pd.Series(bal, env.day_data_df["NESZ"].index[:-1])

    print(qs.stats.sharpe(bal))
    print(qs.stats.cagr(bal))
    