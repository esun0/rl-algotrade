from gym.envs.registration import register
from copy import deepcopy

from . import datasets


register(
    id='forex-v0',
    entry_point='gym_anytrading.envs:ForexEnv',
    kwargs={
        'df': deepcopy(datasets.FOREX_EURUSD_1H_ASK),
        'window_size': 24,
        'frame_bound': (24, len(datasets.FOREX_EURUSD_1H_ASK))
    }
)

register(
    id='stocks-v0',
    entry_point='gym_anytrading.envs:StocksEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)

register(
    id='custom-v0',
    entry_point='gym_anytrading.envs:CustomEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)

register(
    id='daily-v0',
    entry_point='gym_anytrading.envs:DailyEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)

register(
    id='intra-v0',
    entry_point='gym_anytrading.envs:IntraEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)

register(
    id='new_multi_env-v0',
    entry_point='gym_anytrading.envs:NewMultiEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)