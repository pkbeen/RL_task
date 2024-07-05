import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pickle

def get_demo_traj():
    with open('demonstration.pkl', 'rb') as fr:
        demonstration = pickle.load(fr)
    return demonstration
#states actions reward next_states dones


class DQfDNetwork(nn.Module):
    '''
    Pytorch module for Deep Q Network
    '''
    def __init__(self, state_dim, action_dim, hidden_size):
        '''
        Define your Q network architecture here
        '''
        super(DQfDNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        '''
        Get Q values for actions given state
        '''
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    

class ReplayMemory(object):
    def __init__(self, n_step, gamma, buffer_size=50000):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.demonstrations = deque()
        self.buffer = deque(maxlen=self.buffer_size)
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.reward_shaping = 0.01


    def add(self, transition, is_demo = False):
        if is_demo:
            transitions = deque(zip(*transition.values()))
            for demo in transitions:
                self.n_step_buffer.append(demo)
                if len(self.n_step_buffer) == self.n_step:
                    states, actions, rewards, next_states, dones = zip(*self.n_step_buffer)
                    state = states[0]
                    action = actions[0]
                    reward = rewards[0]
                    next_state = next_states[0]
                    done = dones[0]
                    n_reward = sum([self.gamma ** i * reward for i in range(self.n_step)])
                    n_state = next_states[-1]
                    n_done = dones[-1]
                    step_transition = (state, action, self.reward_shaping * reward, next_state, done, self.reward_shaping * n_reward, n_state, n_done, is_demo)
                    self.demonstrations.append(step_transition)
                    if n_done:
                        self.n_step_buffer.clear()
            self.n_step_buffer.clear()

        else:
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) == self.n_step:
                states, actions, rewards, next_states, dones = zip(*self.n_step_buffer)
                state = states[0]
                action = actions[0]
                reward = rewards[0]
                next_state = next_states[0]
                done = dones[0]
                n_reward = sum([self.gamma ** i * reward for i in range(self.n_step)])
                n_state = next_states[-1]
                n_done = dones[-1]
                step_transition = (state, action, self.reward_shaping * reward, next_state, done, self.reward_shaping * n_reward, n_state, n_done, is_demo)
                self.buffer.append(step_transition)
                if n_done:
                    self.n_step_buffer.clear()

    def sample(self, batch_size): #\\
        '''
        samples random batches from buffer
        '''
        sample_buffer = list(self.demonstrations) + list(self.buffer)
        sample = random.sample(sample_buffer, batch_size)

        states, actions, rewards, next_states, dones, n_rewards, n_states, n_dones, is_demo = zip(*sample)

        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.float).unsqueeze(1)
        n_rewards = torch.tensor(np.array(n_rewards), dtype=torch.float).unsqueeze(1)
        n_next_states = torch.tensor(np.array(n_states), dtype=torch.float)
        n_dones = torch.tensor(np.array(n_dones), dtype=torch.float).unsqueeze(1)
        is_demo = torch.tensor(np.array(is_demo), dtype=torch.float).unsqueeze(1)

        return states, actions, rewards, next_states, dones, n_rewards, n_next_states, n_dones, is_demo


class DQfDAgent(object):
    '''
    DQfD train agent
    '''
    def __init__(self, env, state_dim, action_dim):
        # DQN hyperparameters
        self.lr = 0.0005
        self.gamma = 0.99
        self.epsilon = 0.01
        self.eps_decay = 1e-5
        self.eps_min = 0.001
        self.target_update_freq = 20
        self.hidden_size = 128
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.n_step = 4
        self.margin = 0.8
        self.lambda_1 = 1.0
        self.lambda_2 = 1.0
        self.lambda_3 = 1e-5

        self.env = env
        self.main_net = DQfDNetwork(state_dim, action_dim, self.hidden_size)
        self.target_net = DQfDNetwork(state_dim, action_dim, self.hidden_size)
        self.loss_func = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(self.n_step, self.gamma)

    def get_action(self, state):
        '''
        Select actions for given states with epsilon greedy
        '''
        if np.random.uniform() < self.epsilon:
            return np.array([self.env.action_space.sample()])
        else:
            with torch.no_grad():
                q_values = self.main_net(state)
                action = q_values.argmax().numpy()
            return np.array([action])
        

    def calculate_loss(self, mini_batch):
        '''
        Implement DQfD loss function
        '''

        states, actions, rewards, next_states, dones, n_rewards, n_next_states, n_dones, is_demo_ = mini_batch
        # Q values of current states
        q_value = self.main_net(states)
        q_values = self.main_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        one_step_q_loss = self.loss_func(q_values, target_q_values)

        n_step_q_values = self.main_net(states).gather(1, actions)
        with torch.no_grad():
            n_step_next_q_values = self.target_net(n_next_states).max(dim=1, keepdim=True)[0]
            n_step_target_q_values = n_rewards + (1 - n_dones) * (self.gamma**self.n_step) * n_step_next_q_values

        n_step_q_loss = self.loss_func(n_step_q_values, n_step_target_q_values)
        
        expert_mask = torch.zeros_like(q_value)
        expert_mask.scatter_(1, actions, 0)
        margin_loss = is_demo_*((q_value + self.margin*expert_mask).max(1)[0].unsqueeze(1) - q_values)

        l2_reg_loss = torch.tensor(0.0)
        for param in self.main_net.parameters():
            l2_reg_loss += torch.norm(param)


        total_loss = one_step_q_loss + self.lambda_1 *n_step_q_loss + self.lambda_2 *margin_loss.mean() + self.lambda_3 * l2_reg_loss


        return total_loss

    def pretrain(self):
        '''
        DQfD pre-train with the demonstration dataset
        '''
        demonstration = get_demo_traj()
        # Add the demonstration dataset into the replay buffer
        self.memory.add(demonstration, is_demo=True)

        # Pre-train for 1000 steps
        for pretrain_step in range(1000):
            pretrain_batch = self.memory.sample(batch_size=64)
            loss = self.calculate_loss(pretrain_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if pretrain_step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())


    def train(self):
        '''
        DQfD main train function
        '''
        ################################  DO NOT CHANGE  ################################
        episode_scores = deque(maxlen=20)
        mean_scores = []
        train_step = 0
        
        # Pre-train with the demonstration data 
        self.pretrain()       

        for episode in range(250):
            score = 0            
            done = False
            state = self.env.reset()

            while not done:
                action = self.get_action(torch.FloatTensor(state)).item()
                next_state, reward, done, _ = self.env.step(action)
                score += reward 
        #################################################################################
                transition = (state, action, reward, next_state, done)
                self.memory.add(transition)

                batch = self.memory.sample(batch_size=64)
                loss = self.calculate_loss(batch)
                self.epsilon = max(self.eps_min, self.epsilon - self.eps_decay)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if train_step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.main_net.state_dict())

                state = next_state
        ################################  DO NOT CHANGE  ################################
                train_step += 1

                if done:
                    episode_scores.append(score)
                    mean_score = np.mean(episode_scores)
                    mean_scores.append(mean_score)
                    print(f'[Episode {episode}] Avg. score: {mean_score}')

            if (mean_score > 475) and (len(episode_scores) == 20):
                break

        return mean_scores
        #################################################################################

