import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import pickle

from DQN import DQN


def test_already_trained_model(trained_model):
    rewards_list = []
    num_test_episodes = 100
    env = gym.make('LunarLander-v2')
    print("Start testing already trained model")

    step_count = 1000

    for test_episode in range(num_test_episodes):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        total_reward = 0
        for step in range(step_count):
            #env.render()
            action = np.argmax(trained_model.predict(current_state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, num_observation_space])
            current_state = next_state
            total_reward += reward
            if done:
                break
        rewards_list.append(total_reward)
        print("Episode: {}, Reward: {}".format(test_episode, total_reward))

    return rewards_list


def plot_df(df, chart_name, title, x_label, y_label):
    plt.rcParams.update({'font.size': 17})
    df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()

    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)

    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name + '.png')


def plot_df2(df, chart_name, title, x_label, y_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()

    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plt.ylim(y_limit)

    fig = plot.get_figure()
    fig.savefig(chart_name + '.png')


def plot_experiments(df, chart_name, title, x_label, y_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()

    plot = df.plot(linewidth=1, figsize=(15, 8))
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plt.ylim(y_limit)

    fig = plot.get_figure()
    fig.savefig(chart_name + '.png')


def run_experiment_for_gamma():
    print("Start experiment for gamma")
    env = gym.make('LunarLander-v2')

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
    env = gym.make('LunarLander-v2')

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
    env = gym.make('LunarLander-v2')

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


if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode='human')

    env.action_space.seed(21)
    np.random.seed(21)

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    num_episodes = 2000

    print("Start training")
    model = DQN(env, lr, gamma, epsilon, epsilon_decay)
    model.train(num_episodes, True)

    save_dir = "saved_models"
    model.save(save_dir + "trained_model.h5")

    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

    reward_df = pd.DataFrame(rewards_list)
    plot_df(reward_df, 'Rewards form each testing episode', 'Rewards form each testing episode', 'Episode', 'Reward')
    print("Finish training and testing")

    run_experiment_for_gamma()
    run_experiment_for_lr()
    run_experiment_for_epsilon_decay()
