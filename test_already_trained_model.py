import gymnasium as gym


from tensorflow import keras
import numpy as np

def test_already_trained_model():
    trained_model = keras.models.load_model('saved_modelstrained_model.h5')

    rewards_list = []
    num_test_episodes = 10
    env = gym.make('CartPole-v0', render_mode='human')
    print("Start testing already trained model")

    step_count = 1000

    for test_episode in range(num_test_episodes):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state[0], [1, num_observation_space])
        total_reward = 0
        for step in range(step_count):
            # env.render()
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