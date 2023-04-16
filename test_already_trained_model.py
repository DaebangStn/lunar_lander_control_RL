import os.path
import sys

import gymnasium as gym


from tensorflow import keras
import numpy as np

def test_already_trained_model(env=None):
    if env is None:
        env = gym.make('CartPole-v1', render_mode='human')

    if not os.path.exists('saved_models/trained_model.h5'):
        sys.exit("No trained model found, please train the model first")

    print("Load already trained model")
    trained_model = keras.models.load_model('saved_models/trained_model.h5')

    rewards_list = []
    num_test_episodes = 10
    print("Start testing already trained model")

    step_count = 1000

    for test_episode in range(num_test_episodes):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state[0], [1, num_observation_space])
        total_reward = 0
        for step in range(step_count):
            action = np.argmax(trained_model.predict(current_state, verbose=0)[0])
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, num_observation_space])
            current_state = next_state
            total_reward += reward
            if done:
                break
        rewards_list.append(total_reward)
        print("Episode: {}, Reward: {}".format(test_episode, total_reward))

    return rewards_list


if __name__ == '__main__':
    rewards_list = test_already_trained_model()
    print("Average reward: {}".format(sum(rewards_list) / len(rewards_list)))