import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

from .trading_env import TradingEnv #, Actions, Positions


class NewMultiEnv(TradingEnv):

    def __init__(self, df, frame_bound, bankroll, window_size=1, discrete = False, price_visible = False, new_reward=False): #, short=False
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.num_tickers = len(df['prices'])
        super().__init__(df, window_size)
        
        if discrete:
            print('Discrete action space, not implemented')
        else:
            print('Continuous action space')
            
        if price_visible:
            print('Prices visible')
        else:
            print('Prices not visible')
            
        if new_reward:
            print('New reward')
        else:
            print('Standard reward')
        
        self.discrete = discrete
        self.new_reward = new_reward
        self.price_visible = price_visible
        
        self.shape = (2,self.num_tickers+1)
        
        if self.discrete:
            self.action_space = spaces.Discrete(2**(self.num_tickers+1)) # Does not seem optimal
        else:
            self.action_space = spaces.Box(low=0,high=1, shape=(self.num_tickers + 1,))
        
        # Observation space  2 x (num_tickers + 1) first row is current positions
        
        if self.price_visible:
            self.shape = (3,self.num_tickers+1) # last row are last day stock prices

        self.observation_space = spaces.Box(low=0, high=1, shape=self.shape, dtype=np.float32)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        
        self.starting_bankroll=bankroll
        self.net_worth=bankroll
        
        self.positions = np.zeros(self.num_tickers+1)
        self.positions[0] = self.starting_bankroll
        self.directions = self.signal_features
        print('prices shape:', self.prices.shape, 'directions shape:', self.signal_features.shape)
    
    # Override
    def _get_observation(self):
        if self.price_visible:
            last_price = np.insert(self.prices[self._current_tick-1],0,0)
            return np.stack([self.positions, self.directions[self._current_tick-1], last_price])
        return np.stack([self.positions, self.directions[self._current_tick-1]])


    def _process_data(self):
        prices = []
        directions = []
        
        for ticker in self.df['prices']:
            #print(ticker)
            prices.append(self.df['prices'][ticker].loc[:, 'PX_LAST'].to_numpy())
            
        for ticker in self.df['directions']:
            directions.append(self.df['directions'][ticker].to_numpy().mean(axis=1))
        
        prices = np.array(prices).T 
        directions = np.array(directions).T
        
        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        directions = directions[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        directions = np.hstack((np.zeros((directions.shape[0],1)),directions))
        return prices, directions

    def reset(self):
        # Override
        self._done = False
        self._current_tick = self._start_tick
        self.net_worth=self.starting_bankroll
        
        
        self.positions = np.zeros(self.num_tickers+1)
        self.positions[0] = self.starting_bankroll
                                      
        self._position_history = [] 
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}

        return self._get_observation()
    
    def step(self, action):
        self._done = False
        self._current_tick += 1
            
        if self._current_tick == self._end_tick:
            self._done = True
        
        if action.sum() != 0:
            action = action/action.sum()
        else:
            action[0] = 1
               
        new_positions = (action[1:] * self.net_worth) // self.prices[self._current_tick-1]

        # update positions vector
        self.prev_positions = self.positions.copy()

        self.positions[1:] = new_positions

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
        if self.new_reward:
            price_diff = self.prices[self._current_tick] - self.prices[self._current_tick-1]
            r = (self.positions[1:]*price_diff).sum()
            if r < 0:
                r = -(r**2)
            return r
        
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