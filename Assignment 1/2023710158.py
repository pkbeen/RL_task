import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            new_v = 0
            for a, a_p in enumerate(policy[s]):
                for p, s_p, r in env.MDP[s][a]:
                    new_v += a_p *p*(r+gamma*V[s_p])
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for p, s_p, r in env.MDP[s][a]:
                q[a] += p*(r+gamma*V[s_p])
        greedy_a = np.argmax(q)
        policy[s][greedy_a] = 1
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy
    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            q_v = np.zeros(env.nA)
            for a in range(env.nA):
                for p, s_p, r in env.MDP[s][a]:
                    q_v[a] += p * (r + gamma * V[s_p])
            V[s] = np.max(q_v)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q_v = np.zeros(env.nA)
        for a in range(env.nA):
            for p, s_p, r in env.MDP[s][a]:
                q_v[a] += p*(r+gamma*V[s_p])
        best_a = np.argmax(q_v)
        policy[s][best_a] = 1
    return policy, V