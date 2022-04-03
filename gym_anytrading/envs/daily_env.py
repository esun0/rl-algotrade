import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

from .custom_env import CustomEnv #, Actions, Positions


class DailyEnv(CustomEnv):

    def __init__(self, df, window_size, frame_bound, bankroll, dividends):
        super().__init__(df, window_size, frame_bound, bankroll, dividends)