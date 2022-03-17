import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

from .trading_env import TradingEnv #, Actions, Positions


class CustomEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, bankroll, short=False, discrete=False):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.num_tickers = len(df)
        super().__init__(df, window_size)
        
        if discrete:
            print('Discrete action space')
        else:
            print('Continuous action space')
        
        self.shape = (window_size, self.num_tickers, self.signal_features.shape[-1]) # 10 x 4 x 2
        self.action_space = spaces.Box(low=0,high=1, shape=(self.num_tickers + 1,)) #low=-1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        
        self.starting_bankroll=bankroll
        self.net_worth=bankroll
        
        self.short = short
        self.positions = np.zeros(self.num_tickers+1)
        self.positions[0] = self.starting_bankroll
    
    # Override
    '''def _get_observation(self):
        return self.signal_features[:,(self._current_tick-self.window_size):self._current_tick]'''


    def _process_data(self):
        prices = []
        for ticker in self.df:
            print(ticker)
            prices.append(self.df[ticker].loc[:, 'PX_LAST'].to_numpy())
        
        # prices = self.df.loc[:, 'PX_LAST'].to_numpy()
        prices = np.array(prices).T 
        # print(prices.shape)
        
        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices,axis=0), 0, 0, axis=0)
        #print(prices.shape, diff.shape, diff[:3])
        
        signal_features = np.stack((prices, diff),axis=-1) #column_stack (4,2000,2)

        return prices, signal_features

    def reset(self):
        # Override
        self._done = False
        self._current_tick = self._start_tick
        self.net_worth=self.starting_bankroll
        
        
        self.positions = np.zeros(self.num_tickers+1)
        self.positions[0] = self.starting_bankroll
                                      
        self._position_history = [] #(self.window_size * [None]) + [self.positions]
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}
        #print('reset:', self._get_observation().shape)
        return self._get_observation()
    
    def step(self, action):
        # Override
        #print('lol',action.shape)
        self._done = False
        self._current_tick += 1
            
        if self._current_tick == self._end_tick:
            self._done = True
        
        if action.sum() != 0:
            action = action/action.sum()
        else:
            action[0] = 1
        
        #order_volume = action
        '''order_volume =  (action[1:] * self.net_worth) // self.prices[self._current_tick-1]
        
        print(order_volume)
        order_size = order_volume * self.prices[self._current_tick-1]
        
            
        if self.positions[0]<order_size:
            order_volume = self.positions[0]//self.prices[self._current_tick-1] # buys max number of shares using all of its cash
            order_size = order_volume * self.prices[self._current_tick-1]
            #print('insufficient capital for trade: [timestep - {}]'.format(self._current_tick-1))
        
        if not self.short and order_volume+self.positions[1]<0:
            order_volume = -self.positions[1]
            order_size = order_volume * self.prices[self._current_tick-1]'''
        
        new_positions = (action[1:] * self.net_worth) // self.prices[self._current_tick-1]
        #print(action, new_positions)
        # update positions vector
        self.prev_positions = self.positions.copy()
        #self.positions[1] += order_volume
        self.positions[1:] = new_positions
        #self.positions[0] -= order_size
        self.positions[0] = self.net_worth - (new_positions*self.prices[self._current_tick-1]).sum()
        
        step_reward = self._calculate_reward()
        self._total_reward += step_reward

        self._update_profit()
        
        self.net_worth = self.positions[0] + (self.positions[1:]*self.prices[self._current_tick]).sum()
        
        self._position_history.append({'val': self.positions, 'delta': self.positions-self.prev_positions, 'nw': self.net_worth})
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self.positions.copy(),
            delta = self.positions - self.prev_positions,
            net_worth = self.net_worth
        )
        self._update_history(info)

        return observation, step_reward, self._done, info
    
    
    
    def _calculate_reward(self):
        #unrealized
        price_diff = self.prices[self._current_tick] - self.prices[self._current_tick-1]
        return (self.positions[1:]*price_diff).sum()


    def _update_profit(self):
        # unrealized
        price_diff = self.prices[self._current_tick] - self.prices[self._current_tick-1]
        self._total_profit += (self.positions[1:] * price_diff).sum()

                
    def _realized_profit(self, action):
        pass
        
        
    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
    
    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices,'k')
        
        for idx in range(self.num_tickers):
            short_ticks = []
            long_ticks = []
            for i, tick in enumerate(window_ticks):
                if self._position_history[i]['delta'][idx+1] < 0:
                    short_ticks.append(tick)
                elif self._position_history[i]['delta'][idx+1] > 0:
                    long_ticks.append(tick)

            plt.plot(short_ticks, self.prices[short_ticks,idx], 'r.')
            plt.plot(long_ticks, self.prices[long_ticks,idx], 'g.')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
    def render_net_worth(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.history['net_worth'])

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i]['delta'][0] > 0:
                short_ticks.append(tick)
            elif self._position_history[i]['delta'][0] < 0:
                long_ticks.append(tick)

        plt.plot(short_ticks, np.array(self.history['net_worth'])[short_ticks], 'r.')
        plt.plot(long_ticks, np.array(self.history['net_worth'])[long_ticks], 'g.')

        plt.suptitle(
            "Min: %.6f" % min(self.history['net_worth']) + ' ~ ' +
            "Max: %.6f" % max(self.history['net_worth']) + ' ~ ' +
            "End: %.6f" % self.history['net_worth'][-1]
        )