import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None


class TradingEnv():
    """
    Trading environment which allows all-in trades only
    Possible actions are : 1 BUY, 2 SOLD, 0 HOLD
    """

    def __init__(self, data, stock_name="Stock", starting_capital=1000, state_length=50, start_ind=0):
        """

        :param data: OHLC dataset
        :param symbol:
        :param starting_capital:
        :param state_length:
        :param start_ind:
        """
        # panda dataframe
        self.__data = data.copy()

        # current time step
        self.current_time = state_length + start_ind

        self.stock_name = stock_name
        self.starting_capital = starting_capital
        self.state_length = state_length
        self.n_shares = 0

        # Add columns to dataframe
        self.__data.columns = ["date", "open", "high", "low", "close", "adj close", "volume"]
        self.__data['action'] = 0
        self.__data['holdings'] = 0.
        self.__data['cash'] = 0.
        self.__data['cash'][0:self.current_time] = float(starting_capital)
        self.__data['money'] = self.__data['holdings'] + self.__data['cash']
        self.__data['returns'] = 0.

        # reinforcement learning variable
        self.state = np.array([self.__data['close'][start_ind:self.current_time].tolist()])
        self.reward = 0.
        self.done = 0
        self.infos = {}


    def reset(self, start_ind=0):
        """
        reset the trading environment starting at start index
        :param start_ind: int start index
        :return: the initial state
        """
        # reset all the environment
        self.__data['action'] = 0
        self.__data['holdings'] = 0.
        self.__data['cash'] = 0.
        self.__data['cash'][0:self.state_length + start_ind] = self.starting_capital
        self.__data['money'] = self.__data['holdings'] + self.__data['cash']
        self.__data['returns'] = 0.
        self.n_shares = 0
        self.current_time = start_ind + self.state_length
        self.state = (np.array((self.__data['close'][start_ind:self.state_length + start_ind].tolist())).reshape(-1, 1)).flatten()
        self.reward = 0.
        self.done = 0
        return self.state

    def step(self, action):
        """
        perform the given action and update the environment to the next trading time step
        :param action: int (0-HOLD, 1-BUY, 2-SELL)
        :return: next_state, reward, done, infos
        """

        prev_time = self.current_time - 1

        # BUY
        if action == 1:
            if self.n_shares != 0:
                print("\033[93m [WARNING] you can not perform this action. Choose between 0 to HOLD, 2 to SOLD")
                return self.state, self.reward, self.done, self.infos
            self.__data["action"][prev_time] = 1
            self.n_shares += self.__data["cash"][prev_time] / self.__data["close"][prev_time]  # all in
            self.__data["cash"][prev_time + 1] = 0
            self.__data["holdings"][prev_time + 1] = self.n_shares * self.__data["close"][prev_time + 1]

        # HOLD
        elif action == 0:
            self.__data["action"][prev_time] = 0
            self.__data['cash'][prev_time + 1] = self.__data['cash'][prev_time]
            self.__data["holdings"][prev_time + 1] = self.n_shares * self.__data["close"][prev_time + 1]


        # SOLD
        elif action == 2:
            if self.n_shares == 0:
                print("\033[93m [WARNING] you can not perform this action. Choose between 0 to HOLD, or 1 to BUY")
                return self.state, self.reward, self.done, self.infos
            self.__data["action"][prev_time] = 2
            self.__data["cash"][prev_time + 1] = self.__data["cash"][prev_time] + self.n_shares * self.__data["close"][prev_time]
            self.__data["holdings"][prev_time + 1] = 0
            self.n_shares = 0

        # Update the total amount of money and the daily return
        self.__data['money'][prev_time + 1] = self.__data['holdings'][prev_time + 1] + self.__data['cash'][prev_time + 1]
        self.__data['returns'][prev_time + 1] = (self.__data['money'][prev_time + 1] - self.__data['money'][prev_time]) / self.__data['money'][prev_time]

        # compute the reward
        self.reward = (self.__data['money'][prev_time + 1] - self.__data["money"][prev_time]) / self.__data["money"][prev_time]

        # update time step
        self.current_time = self.current_time + 1

        # update state
        self.state = (np.array(self.__data['close'][self.current_time - self.state_length: self.current_time].tolist()).reshape(-1, 1)).flatten()

        # END of the simulation
        if self.current_time == self.__data.shape[0] or self.__data["money"][prev_time] == 0:
            self.done = 1

        return self.state, self.reward, self.done, self.infos


    def render(self, title=""):
        """
        plots price evolution of the stock market  and evolution of trading capital
        :param title: Graph title
        """

        fig = plt.figure(figsize=(10, 8))

        ax1 = fig.add_subplot(211, ylabel=f'{self.stock_name} Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)


        # plotting the stock price evolution
        self.__data['close'][self.state_length:self.current_time].plot(ax=ax1, color='blue', lw=2)

        # plotting the capital evolution
        self.__data['money'][self.state_length:self.current_time].plot(ax=ax2, color='blue', lw=2)

        # plotting the action taken during the simulation
        for ax, name in zip([ax1, ax2], ["close", "money"]):

            ax.plot(self.__data.loc[self.__data['action'] == 1].index,
                    self.__data[name][self.__data['action'] == 1],
                 '^', markersize=5, color='green')
            ax.plot(self.__data.loc[self.__data['action'] == 2].index,
                    self.__data[name][self.__data['action'] == 2],
                 'v', markersize=5, color='red')

        # Title and Legend
        ax1.title.set_text(title)
        ax1.legend(["Price", "BUY", "SOLD"])
        ax2.legend(["Capital", "BUY", "SOLD"])
        plt.show()


    def get_sharpe_ratio(self):
        """
        :return: Sharpe ratio
        """
        return np.mean(self.__data["returns"][self.state_length:self.current_time - 1]) / np.std(self.__data["returns"][self.state_length:self.current_time - 1]) * (252 ** 0.5)

    def get_cum_return(self):
        """
        :return: cumulative return
        """
        return (self.__data["money"][self.current_time - 1] - self.__data["money"][0]) / self.__data["money"][0]

    def __repr__(self):
        return str(self.state)

    def get_all_data(self):
        return self.__data

    def get_possible_actions(self):
        return [0, 1, 2]