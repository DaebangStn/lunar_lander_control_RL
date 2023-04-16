import gymnasium as gym

import os
import numpy as np

import pandas as pd
import pickle
import tensorflow as tf

from agents.DQN import DQN
from plots import plot_df
from test_already_trained_model import test_already_trained_model


if __name__ == '__main__':
    print(f'\n{tf.config.list_physical_devices("GPU")}\n')

    gym.envs.register(
        id='CartPole-v1-reward-1',
        entry_point='envs.CartPole_rewards:CartPole_reward_1',
        max_episode_steps=1000,
    )

    #env = gym.make('CartPole-v1-reward-1')
    env = gym.make('CartPole-v1-reward-1', render_mode='human')
    #env = gym.make('CartPole-v1')
    #env = gym.make('CartPole-v1', render_mode='human')

    env.action_space.seed(21)
    np.random.seed(21)

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    num_episodes = 100

    print("Start training")
    model = DQN(env, lr, gamma, epsilon, epsilon_decay)
    model.train(num_episodes, True)

    # check if there is folder named temp, if not, create the folder
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    save_dir = "saved_models"
    model.save_model(save_dir + "/" + "_trained_model.h5")

    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

    reward_df = pd.DataFrame(rewards_list)
    plot_df(reward_df, 'Rewards form each testing episode', 'Rewards form each testing episode', 'Episode', 'Reward')
    print("Finish training and testing")

    test_already_trained_model(env)

'''
    run_experiment_for_gamma()
    run_experiment_for_lr()
    run_experiment_for_epsilon_decay()
'''
