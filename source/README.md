# Stock Trading Strategy Using Deep Reinforcement Learning and Genetic Algorithm
### Resume 
Stock trading involves buying and selling stocks frequently in an attempt to time the market. It is a very challenging task to design a profitable strategy.
Investors who trade stocks do extensive research, often devoting hours a day to following the market. 
They rely on technical stock analysis, using tools to chart a stock's movements in an attempt to find trading opportunities and trends. 
We explore an innovative approach based on deep reinforcement learning and genetic algorithm to build trading agents which can manage portfolios.

### Files Description
trading_env.py:
This files contains one class, TradingEnv, which represent the trading environment.

agents.py:  
This files contains several class, each representing a trading agent. 
It is easy to add new agent by creating a new class which inherit from the TradingAgent parent class.

GeneticAlgo.py:
This file implement several function (not implemented in OOP) to use genetic algorithm for finding the best parameters for our trading agents.
The main function is gen_algo()

load_data.py:
this file is composed of a unique function which read csv file from the /data directory and return a pandas dataframe.

 