from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import tensorflow as tf
from tqdm import tqdm
from sklearn import preprocessing
from tensorflow import keras


class TradingAgent(ABC):
    """
    Abstract class for trading agent
    """
    def __init__(self):
        self.actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
        # using this variable to update the legal actions. each agent must update it if and only
        # if it perform a BUY action or a SELL action
        self.last_action = 0


    def get_legal_actions(self):
        """
        return a list with all legal actions.
        an agent can only perform one trade at a time which means that if it holds stock it cannot buy again.
        Obviously, the agent cannot sell if it doesn't hold any holdings
        """
        if self.last_action == 1:
            return [2, 0]
        elif self.last_action == 0:
            return[0, 1]
        else:
            return [1, 0]

    @abstractmethod
    def get_params(self):
        """
        return the  list of  parameters which influences the policy of the agent ( used for genetic algorithm)
        """
        pass

    @abstractmethod
    def get_action(self, state):
        """
        given a state the agent return an action: 0 - HOLD, 1 - BUY, 2 - SELL
        :param state: the current state
        :return: int action
        """
        pass

    @abstractmethod
    def reset(self):
        """
        reset the agent
        """
        pass

class BHAgent(TradingAgent):
    """
    Buy and Hold agent
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        if 1 in self.get_legal_actions():
            self.last_action = 1
            return 1  # buy
        else:
            return 0  # hold

    def get_params(self):
        return []

    def reset(self):
        self.last_action = 0
        return self


class RSIAgent(TradingAgent):
    """
     Relative strength index agent
     RSI = 100.0 - (100.0 / (1.0 + RS))
     with RS = GAIN EMA / LOSS EMA
     the strategy will be:
     IF CURRENT RSI < low_threshold ==> BUY SIGNAL
     IF CURRENT RSI > high_threshold ==> SELL SIGNAL
    """
    def __init__(self, period=14, low_threshold=30, high_threshold=70):
        """

        :param period: window period to use for calculating EMA
        :param low_threshold: int between 0-100
        :param high_threshold: int between 0-100
        """
        super().__init__()
        self.period = period
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.k = 2 / (self.period + 1)
        self.up = []
        self.down = []
        self.loss_ema = None
        self.gain_ema = None
        self.all_rsi = []

    def reset(self):
        self.up = []
        self.down = []
        self.loss_ema = None
        self.gain_ema = None
        self.all_rsi = []
        self.last_action = 0
        return self

    def get_params(self):
        return [self.period, self.low_threshold, self.high_threshold]

    def __repr__(self):
        return f"RSI Agent with parameters: period: {self.period}, low_threshold: {self.low_threshold}, high_threshold: {self.high_threshold}"

    def get_action(self, state):
        """
        return the best legal move
        :param state: the current state
        """
        action = self.get_best_move(state)
        if action in self.get_legal_actions():
            if action in [1, 2]:
                self.last_action = action
            return action
        else:
            return 0

    def get_best_move(self, state):
        """
        return the best move without checking legality
        :param state: the current state
        """

        # starting case
        if len(self.up) == 0:
            # computing gain and losses
            ret = np.diff(state)
            for i in range(len(ret)):
                if ret[i] < 0:
                    self.up.append(0)
                    self.down.append(abs(ret[i]))
                else:
                    self.up.append(ret[i])
                    self.down.append(0)
            # initial loss and gain EMA ( exponential moving average)
            self.loss_ema = np.mean(self.down[-self.period:])
            self.gain_ema = np.mean(self.up[-self.period:])
            return 0  # HOLD

        # computing the last difference
        ret = state[-1] - state[-2]
        if ret >= 0:  # gain
            self.up.append(ret)
            self.down.append(0)
        else:
            self.up.append(0)
            self.down.append(abs(ret))
        # computing the loss and gain EMA
        self.loss_ema = self.down[-1] * self.k + self.loss_ema * (1 - self.k)
        self.gain_ema = self.up[-1] * self.k + self.gain_ema * (1 - self.k)

        rs = self.gain_ema/self.loss_ema
        rsi = 100 - (100 / (1 + rs))

        self.all_rsi.append(rsi)

        # strategy
        if len(self.all_rsi) < 2:
             return 0

        rsi = self.all_rsi[-1]

        # BUY
        if rsi < self.low_threshold:
             return 1

        # SELL
        elif rsi > self.high_threshold:
            return 2

        # HOLD
        return 0


class MACDAgent(TradingAgent):
    """
    This agent is using the MACD indicator to take action
    The MACD is define as SHORT EMA - LONG EMA
    when EMA with period n of a dataset d: EMAtoday(d,n) = d[i] * 2/(n+1) + EMAyesterday(d,n) * (1 - 2/(n+1))
    the strategy is :
    IF MACD LINE > SIGNAL LINE => BUY THE STOCK
    IF SIGNAL LINE > MACD LINE => SELL THE STOCK
    """
    def __init__(self, s_period=12, l_period=26, signal_period=9):
        """
        :param s_period: short period window for computing the EMA
        :param l_period: long period window for computing the EMA
        :param signal_period: The signal period window for the MACD EMA (signal line)
        """
        super().__init__()
        self.s_period = s_period
        self.l_period = l_period
        self.signal_period = signal_period
        self.lema = None
        self.sema = None
        self.kl = 2 / (self.l_period + 1)
        self.ks = 2 / (self.s_period + 1)
        self.k_signal = 2 / (self.signal_period + 1)
        self.all_macd = []
        self.last_action = 0
        self.signal = None

    def get_action(self, state):
        """
        return the best legal action
        :param state: current state
        """
        action = self.get_best_move(state)
        if action in self.get_legal_actions():
            if action in [1, 2]:
                self.last_action = action
            return action
        else:
            return 0

    def get_best_move(self, state):
        """
        return the best move without checking legality
        :param state: current state
        """
        if self.sema is None:
            self.sema = np.mean(state[-self.s_period:-1])
            self.lema = np.mean(state[-self.l_period:-1])

        # calculating EMA
        self.sema = state[-1] * self.ks + self.sema * (1 - self.ks)
        self.lema = state[-1] * self.kl + self.lema * (1 - self.kl)

        # calculating MACD
        self.all_macd.append(self.sema - self.lema)


        # signal ligne (by default 9 EMA of the macd)
        if len(self.all_macd) >= self.signal_period:
            if self.signal is None:
                self.signal = np.mean(self.all_macd[-self.signal_period:])
            self.signal = self.all_macd[-1] * self.k_signal + self.signal * (1 -self.k_signal)


        # taking a decision
        if self.signal is None:
            return 0
        if self.all_macd[-1] > self.signal:
            return 1
        elif self.signal > self.all_macd[-1]:
            return 2

        return 0

    def __repr__(self):
        return f"MACD agent with parameters: short period: {self.s_period}, long period: {self.l_period}, signal period: {self.signal_period}"

    def get_params(self):
        return [self.s_period, self.l_period, self.signal_period]

    def reset(self):
        """
        reset the agent
        :return: the agent
        """
        self.sema = None
        self.lema = None
        self.all_macd = []
        self.last_action = 0
        self.signal = None
        return self


class RSI_MACD_Agent(TradingAgent):
    """
    Mix RSI and MACD strategy
    """
    def __init__(self, rsi_window=14, rsi_low=30, rsi_high=70, macd_short=12, macd_long=26, macd_signal=9):
        """
        :param rsi_window: window period to use for calculating rsi EMA
        :param rsi_low:  rsi low threshold (int between 0-100)
        :param rsi_high: rsi high threshold (int between 0-100)
        :param macd_short: macd short period EMA
        :param macd_long: macd long period EMA
        :param macd_signal: macd signal EMA
        """
        super().__init__()
        self.rsi_window = rsi_window
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.macd_signal = macd_signal
        self.rsi_agent = RSIAgent(self.rsi_window, self.rsi_low, self.rsi_high)
        self.macd_agent = MACDAgent(self.macd_short, self.macd_long, self.macd_signal)

    def get_params(self):
        return self.rsi_agent.get_params() + self.macd_agent.get_params()

    def reset(self):
        self.rsi_agent.reset()
        self.macd_agent.reset()
        self.last_action = 0
        return self

    def __repr__(self):
        return f"RSI + MACD: rsi period: {self.rsi_window}, rsi low_threshold: {self.rsi_low}, rsi high_threshold: {self.rsi_high}" \
               f" macd short period: {self.macd_short}, macd long period: {self.macd_long}, macd signal period: {self.macd_signal}"

    def get_action(self, state):
        action1 = self.rsi_agent.get_best_move(state)
        action2 = self.macd_agent.get_best_move(state)

        # av_action = 0.5 * action1 + 0.5 * action2
        if action1 == action2 and action1 in self.get_legal_actions():
            if action1 in [1, 2]:
                self.last_action = action1
            return action1

        return 0


class DQNAgent(TradingAgent):
    """
    Use Deep Q learning algorithm
    """

    def __init__(self, obs_shape, n_actions, epsilon=0.4, discount_rate=0.95, batch_size=32, lr=1e-3,
             load_model=None):
        super().__init__()

        # input to our neural network
        self.obs_shape = obs_shape

        # 3 for trading case
        self.n_actions = n_actions

        # trained model
        if load_model is not None:
            self.model = keras.models.load_model(load_model)
        else:
            self.model = self.__init_nn()

        # replay memory used to create batch for training
        self.replay_memory = deque(maxlen=500)

        # DQL variables
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = keras.optimizers.Adam(lr=self.lr)
        self.loss = keras.losses.mean_squared_error
        self.min_max_scaler = preprocessing.MinMaxScaler()

    def __init_nn(self):
        """
        initialise the neural netowrk architecture
        :return: tensorflow model
        """
        model = keras.Sequential([
            # keras.layers.Flatten(),
            keras.layers.LayerNormalization(),
            keras.layers.Dense(64, activation="relu", input_shape=(self.obs_shape,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(3)
        ])
        # print(model.summary())
        return model

    def sample_from_replay_mem(self):
        """
        sample a batch from the replay memory
        :return: states -> [batch_size, observation_space], actions -> [batch_size, n_actions],
        rewards -> [batch_size, 1], next_states -> [batch_size, observation_space], dones -> [batch_size, 1]
        """
        ind = np.random.randint(len(self.replay_memory), size=self.batch_size)
        batch = [self.replay_memory[i] for i in ind]
        states, actions, rewards, next_states, dones = [np.array([replay[i] for replay in batch]) for i in range(5)]
        return states, actions, rewards, next_states, dones

    def play_one_training_step(self, env, state):
        """
        play one training step, which is choose an action with exploration, and save step into replay memory
        :return:
        """
        action = self.get_action(state, exploration=True)
        next_state, reward, done, info = env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def training_step(self):
        """
        train the neural network to approximate the q values
        :return: the loss (mean square error)
        """
        states, actions, rewards, next_states, dones = self.sample_from_replay_mem()
        next_q = self.model.predict(next_states)
        max_next_q = np.max(next_q, axis=1)
        target_q = rewards + (1 - dones) * self.discount_rate * max_next_q
        target_q = target_q.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_actions)
        with tf.GradientTape() as tape:
            all_q = self.model(states)
            q_val = tf.reduce_sum(all_q * mask, axis=1, keepdims=True)
            loss = tf.reduce_sum(self.loss(target_q, q_val))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def play_validation(self, val_env, step=None):
        """
        used to compute the validation score on a validation environment
        """
        done = 0
        state = val_env.reset()
        i = 0
        while not done:
            i += 1
            action = self.get_action(state, exploration=False)
            state, reward, done, info = val_env.step(action)
            if step is not None:
                if i >= step:
                    break
        return val_env.get_all_data()["money"][len(val_env.get_all_data()) - 1]

    def fit(self, env, episodes=100, val_env=None, save_model=None):
        """
        call this function to train the agent on the given environment
        :param env: environment
        :param episodes: number of episodes to train on
        :param val_env: if not None, used a validation environment to evaluate the model
        :param save_model: if not None, save the model to models/save_model
        """

        best_score = 0
        best_weights = None
        loss = 1
        validation_score = 0
        for ep in tqdm(range(episodes)):
            start = np.random.randint(0, len(env.get_all_data()) - env.state_length)
            obs = env.reset(start_ind=start)
            self.last_action = 0
            self.epsilon = max(1 - ep / 500, 0.01)
            for step in range(200):
                obs, rewards, done, info = self.play_one_training_step(env, obs)
                if done:
                    break

            if ep > 5:
                loss = self.training_step()
                if val_env and ep % 10 == 0:
                    self.last_action = 0
                    validation_score = self.play_validation(val_env)
                    print(f"validation_score: {validation_score}")
                if best_score < validation_score:
                    best_score = validation_score
                    best_weights = self.model.get_weights()

            print(f"episode: {ep}, capital:{env.get_all_data()['money'][env.current_time - 1]}, loss: {loss}")

        self.model.set_weights(best_weights)
        if save_model:
            self.model.save("models/" + save_model)

    def get_action(self, state, exploration=False):
        """
        get the best legal action
        :param state:
        :param exploration: if True, take a random action with probability epsilon
        """
        if exploration and np.random.rand() < self.epsilon:

            action = np.random.choice([0, 1, 2])
            if action not in self.get_legal_actions():
                x = [0, 1, 2]
                x.remove(action)
                action = np.random.choice(x)
            if action in [1, 2]:
                self.last_action = action
            return action
        else:
            q_values = self.model.predict(np.expand_dims(state, 0))[0]
            actions_ind = np.argsort(q_values)
            action = actions_ind[-1]
            if action not in self.get_legal_actions():
                action = 0
            if action in [1, 2]:
                self.last_action = action
            return action

    def get_params(self):
        pass

    def reset(self):
        self.last_action = 0

