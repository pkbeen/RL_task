import gym
import pybullet_envs
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0] #5
ACT_DIM = ENV.action_space.shape[0] #1
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()

#########################################################################################################################
############ ì´ templateì—ì„œëŠ” DO NOT CHANGE ë¶€ë¶„ì„ ì œì™¸í•˜ê³  ë§ˆìŒëŒ€ë¡œ ìˆ˜ì •, êµ¬í˜„ í•˜ì‹œë©´ ë©ë‹ˆë‹¤                    ############
#########################################################################################################################

## ì£¼ì˜ : "InvertedPendulumSwingupBulletEnv-v0"ì€ continuious action space ìž…ë‹ˆë‹¤.
## Asynchronous Advantage Actor-Critic(A3C)ë¥¼ ì°¸ê³ í•˜ë©´ ë„ì›€ì´ ë  ê²ƒ ìž…ë‹ˆë‹¤.

class NstepBuffer:
    '''
    Save n-step transitions to buffer
    '''
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        '''
        Add a sample to the buffer
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self):
        '''
        Sample transitions from the buffer
        '''
        return self.states, self.actions, self.rewards, self.next_states, self.dones

    def reset(self):
        '''
        Reset the buffer
        '''
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.a1 = nn.Linear(OBS_DIM, 32)
        self.a2 = nn.Linear(32, 32)
        self.mu = nn.Linear(32, ACT_DIM)
        self.sigma = nn.Linear(32, ACT_DIM)
        self.c1 = nn.Linear(OBS_DIM, 32)
        self.c2 = nn.Linear(32, 32)
        self.v = nn.Linear(32, ACT_DIM)
        self.set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def actor(self, states):
        a1 = torch.tanh(self.a1(states))
        a2 = torch.tanh(self.a2(a1))
        mean = self.mu(a2)
        std = F.softplus(self.sigma(a2)) + 0.001    #to avoid 0
        return mean, std
    
    def critic(self, states):
        c1 = torch.tanh(self.c1(states))
        c2 = torch.tanh(self.c2(c1))
        values = self.v(c2)
        return values

class Worker(object):
    def __init__(self, global_actor, global_epi, sync, finish, n_step, seed):
        self.env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
        self.env.seed(seed)
        self.lr = 0.00045
        self.gamma = 0.95
        self.entropy_coef = 0.01
        ############################################## DO NOT CHANGE ##############################################
        self.global_actor = global_actor
        self.global_epi = global_epi
        self.sync = sync
        self.finish = finish
        self.optimizer = optim.Adam(self.global_actor.parameters(), lr=self.lr)
        ###########################################################################################################  
        
        self.n_step = n_step
        self.local_actor = ActorCritic()
        self.nstep_buffer = NstepBuffer()

    
    def select_action(self, state):
        '''
        selects action given state

        return:
            continuous action value
        '''
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.local_actor.actor(state)
        action_dist = Normal(mean, std)
        action = action_dist.sample()
        return action

    def train_network(self, states, actions, rewards, next_states, dones):
        '''
        Advantage Actor-Critic training algorithm
        '''
        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Convert numpy arrays to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)



        # Compute target values
        with torch.no_grad():
            next_state_value = self.local_actor.critic(next_states)
            target_values = rewards + self.gamma * (1 - dones) * next_state_value
        advantages = target_values - self.local_actor.critic(states)

        # Actor loss
        mean, std = self.local_actor.actor(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions.unsqueeze(1))
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_values = self.local_actor.critic(states)
        critic_loss = F.mse_loss(critic_values, target_values.detach())

        # Entropy loss
        entropy_loss = (dist.entropy().mean())
        total_loss = actor_loss + 0.5*critic_loss - self.entropy_coef * entropy_loss
        nn.utils.clip_grad_norm_(self.global_actor.parameters(), 0.5)

        ############################################## DO NOT CHANGE ##############################################
        # Global optimizer update ì¤€ë¹„
        self.optimizer.zero_grad()
        total_loss.backward()

        # Local parameterë¥¼ global parameterë¡œ ì „ë‹¬
        for global_param, local_param in zip(self.global_actor.parameters(), self.local_actor.parameters()):
                global_param._grad = local_param.grad

        # Global optimizer update
        self.optimizer.step()

        # Global parameterë¥¼ local parameterë¡œ ì „ë‹¬
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        ###########################################################################################################  
    def train(self):
        step = 1

        while True:
            state = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.nstep_buffer.add(state, action.item(), reward, next_state, done)
                # n step에 도달하거나 에피소드가 종료된 경우 학습 진행
                if step % self.n_step == 0 or done:
                    self.train_network(*self.nstep_buffer.sample())
                    self.nstep_buffer.reset()
                state = next_state
                step += 1
                   
            ############################################## DO NOT CHANGE ##############################################
            # ì—í”¼ì†Œë“œ ì¹´ìš´íŠ¸ 1 ì¦ê°€                
            with self.global_epi.get_lock():
                self.global_epi.value += 1
            
            # evaluation ì¢…ë£Œ ì¡°ê±´ ë‹¬ì„± ì‹œ local process ì¢…ë£Œ
            if self.finish.value == 1:
                break

            # ë§¤ ì—í”¼ì†Œë“œë§ˆë‹¤ global actorì˜ evaluationì´ ëë‚  ë•Œê¹Œì§€ ëŒ€ê¸° (evaluation ë„ì¤‘ íŒŒë¼ë¯¸í„° ë³€í™” ë°©ì§€)
            with self.sync:
                self.sync.wait()
            ###########################################################################################################
        self.env.close()