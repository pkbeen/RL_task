import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, Q, mode, eps=1.0, eps_decay=0.01, eps_min=0.01):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        if mode == "mc_control":
            self.alpha = 0.1        #얼마나 새로운거를 받아들일건지 -> alpha가 클 수록 새로운거를 많이 받아들인다는 의미
            self.gamma = 0.995     #현재 state에서 멀어질 수록 reward를 받는 비율을 얼만큼 할거냐? -> 클 수록 많이 받는다 -> 멀리본다.
            self.returns = list()
        elif mode == "q_learning":
            self.alpha = 0.1
            self.gamma = 0.99


    def select_action(self, state):
        if self.mode == "test_mode":
            action = np.argmax(self.Q[state])
        else:
            if np.random.rand() < self.eps:
                action = np.random.randint(len(self.Q[state]))
            else:
                action = np.argmax(self.Q[state])
        return action

    def step(self, state, action, reward, next_state, done):
        if self.mode == "mc_control":
            self.mc_control(state, action, reward, next_state, done)
        elif self.mode == "q_learning":
            self.q_learning(state, action, reward, next_state, done)
        if not done:#exploitation부분을 위한 epsilon 갱신
            self.eps = max(self.eps_min, self.eps - self.eps_decay) #while문으로도 작성 가능하지만 max를 사용하면 계산 속도가 빨라짐.

    def mc_control(self, state, action, reward, next_state, done):
        if done:
            G = defaultdict(lambda: np.zeros(self.n_actions))
            for history in reversed(self.returns):
                state, action, reward = history
                G[state][action] = reward + self.gamma * G[state][action]
                self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha*(G[state][action])
            self.returns.clear()
        else:
            self.returns.append((state, action, reward))      


    def q_learning(self, state, action, reward, next_state, done):
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
