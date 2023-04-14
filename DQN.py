import numpy as np
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
from keras.losses import mean_squared_error


class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []

        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.num_action_space = self.action_space.n
        self.num_observation_space = self.observation_space.shape[0]
        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.num_action_space, activation=linear))

        model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_action_space)

        predicted_action = self.model.predict(state, verbose=0)
        return np.argmax(predicted_action[0])

    def add_to_reply_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def learn_and_update_weights_by_replay(self):
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return

        # Early stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        random_sample = self.get_random_sample_from_replay_memory()
        state, action, rewards, next_states, done_list = self.get_attributes_from_sample(random_sample)

        targets = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - done_list)
        target_vec = self.model.predict_on_batch(state)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [action]] = targets

        self.model.fit(state, target_vec, epochs=1, verbose=0)

    def get_attributes_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        action = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        return np.squeeze(states), action, rewards, next_states, done_list

    def get_random_sample_from_replay_memory(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)

    def train(self, num_episodes=2000, can_stop=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            reward_for_episode = 0
            num_steps = 1000
            state = np.reshape(state[0], [1, self.num_observation_space])

            for step in range(num_steps):
                #self.env.render()
                received_action = self.get_action(state)
                #print("Received action: {}".format(received_action))

                next_state, reward, done, _, _ = self.env.step(received_action)
                next_state = np.reshape(next_state, [1, self.num_observation_space])

                self.add_to_reply_memory(state, received_action, reward, next_state, done)
                reward_for_episode += reward
                state = next_state
                self.update_counter()

                self.learn_and_update_weights_by_replay()

                if done:
                    break

            self.rewards_list.append(reward_for_episode)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            last_rewards_mem = np.mean(self.rewards_list[-100:])
            if last_rewards_mem > 200 and can_stop:
                print("Solved after {} episodes".format(episode))
                print("DQN training finished")
                break
            print("Episode: {}, Reward: {}, Epsilon: {}".format(episode, reward_for_episode, self.epsilon))

    def update_counter(self):
        self.counter = (self.counter + 1) % 5

    def save_model(self, file_name):
        self.model.save(file_name)
