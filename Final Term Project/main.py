import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
from agent import DQfDAgent

########################################################################################
############                                                                ############
############      학습 및 평가 부분(목표 reward에 도달할 때의 episode확인)     ############
############                                                                ############
########################################################################################
def main():
    # 환경 선언 
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 시드 고정
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    env.seed(1)
    env.action_space.seed(1)
    
    # DQfDagent 선언
    dqfd_agent = DQfDAgent(env, state_dim, action_dim)
    
    # DQfD agent train
    print('START train')
    mean_scores = dqfd_agent.train()
    print('END train')
    env.close()

    plt.plot(np.arange(len(mean_scores)), mean_scores)
        
    plt.title('DQfD result')
    plt.xlim(0, 250)
    plt.ylim(0, 500)
    plt.xlabel('Episode')
    plt.ylabel('Average of the last 20 episodes rewards')
    plt.savefig('plot.png')
    plt.close()    

if __name__ == "__main__":
    main()
    