import gymnasium as gym

import numpy as np

import pandas as pd
import pickle
import tensorflow as tf

from DQN import DQN
from plots import plot_df
from run_experiments import run_experiment_for_gamma, run_experiment_for_lr, run_experiment_for_epsilon_decay
from test_already_trained_model import test_already_trained_model


if __name__ == '__main__':
    print(f'\n{tf.config.list_physical_devices("GPU")}\n')
    env = gym.make('CartPole-v0', render_mode='human')

    env.action_space.seed(21)
    np.random.seed(21)

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.997
    gamma = 0.99
    num_episodes = 700

    print("Start training")
    model = DQN(env, lr, gamma, epsilon, epsilon_decay)
    model.train(num_episodes, True)

    save_dir = "saved_models"
    model.save_model(save_dir + "_trained_model.h5")

    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

    reward_df = pd.DataFrame(rewards_list)
    plot_df(reward_df, 'Rewards form each testing episode', 'Rewards form each testing episode', 'Episode', 'Reward')
    print("Finish training and testing")

    test_already_trained_model()

'''
    run_experiment_for_gamma()
    run_experiment_for_lr()
    run_experiment_for_epsilon_decay()
'''
