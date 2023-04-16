import gymnasium as gym

import numpy as np

import pandas as pd
import pickle

from agents.DQN import DQN
from plots import plot_experiments

def run_experiment_for_gamma():
    print("Start experiment for gamma")
    env = gym.make('CartPole-v0')

    env.action_space.seed(21)
    np.random.seed(21)

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma_list = [0.99, 0.9, 0.8, 0.7]
    num_episodes = 1000

    rewards_list_for_gamma = []
    for gamma in gamma_list:
        model = DQN(env, lr, gamma, epsilon, epsilon_decay)
        print("Start training for gamma: {}".format(gamma))
        model.train(num_episodes, False)
        rewards_list_for_gamma.append(model.rewards_list)

    pickle.dump(rewards_list_for_gamma, open("rewards_list_for_gamma.p", "wb"))
    rewards_list_for_gamma = pickle.load(open("rewards_list_for_gamma.p", "rb"))

    gamma_rewards_pd = pd.DataFrame(index=pd.Series(range(1, num_episodes + 1)))
    for i in range(len(gamma_list)):
        gamma_rewards_pd['gamma_' + str(gamma_list[i])] = rewards_list_for_gamma[i]
    plot_experiments(gamma_rewards_pd, 'gamma_rewards', 'Rewards for different gamma values',
                     'Episode', 'Reward', (-600, 300))


def run_experiment_for_lr():
    print("Runnning experiment for lr")
    env = gym.make('CartPole-v0')

    env.action_space.seed(21)
    np.random.seed(21)

    lr_list = [0.001, 0.0001, 0.00001]
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    tranning_episodes = 1000
    rewards_list_for_lr = []

    for lr in lr_list:
        model = DQN(env, lr, gamma, epsilon, epsilon_decay)
        print("Start training for lr: {}".format(lr))
        model.train(tranning_episodes, False)
        rewards_list_for_lr.append(model.rewards_list)

    pickle.dump(rewards_list_for_lr, open("rewards_list_for_lr.p", "wb"))
    rewards_list_for_lr = pickle.load(open("rewards_list_for_lr.p", "rb"))

    lr_rewards_pd = pd.DataFrame(index=pd.Series(range(1, tranning_episodes + 1)))
    for i in range(len(lr_list)):
        lr_rewards_pd['lr_' + str(lr_list[i])] = rewards_list_for_lr[i]
    plot_experiments(lr_rewards_pd, 'lr_rewards', 'Rewards for different lr values',
                     'Episode', 'Reward', (-2000, 300))

def run_experiment_for_epsilon_decay():
    print("Runnning experiment for epsilon_decay")
    env = gym.make('CartPole-v0')

    env.action_space.seed(21)
    np.random.seed(21)

    lr = 0.001
    epsilon = 1.0
    epsilon_decay_list = [0.999, 0.995, 0.99, 0.9]
    gamma = 0.99
    tranning_episodes = 1000
    rewards_list_for_epsilon_decay = []

    for epsilon_decay in epsilon_decay_list:
        model = DQN(env, lr, gamma, epsilon, epsilon_decay)
        print("Start training for epsilon_decay: {}".format(epsilon_decay))
        model.train(tranning_episodes, False)
        rewards_list_for_epsilon_decay.append(model.rewards_list)

    pickle.dump(rewards_list_for_epsilon_decay, open("rewards_list_for_epsilon_decay.p", "wb"))
    rewards_list_for_epsilon_decay = pickle.load(open("rewards_list_for_epsilon_decay.p", "rb"))

    epsilon_decay_rewards_pd = pd.DataFrame(index=pd.Series(range(1, tranning_episodes + 1)))
    for i in range(len(epsilon_decay_list)):
        epsilon_decay_rewards_pd['epsilon_decay_' + str(epsilon_decay_list[i])] = rewards_list_for_epsilon_decay[i]
    plot_experiments(epsilon_decay_rewards_pd, 'epsilon_decay_rewards', 'Rewards for different epsilon_decay values',
                     'Episode', 'Reward', (-600, 300))
