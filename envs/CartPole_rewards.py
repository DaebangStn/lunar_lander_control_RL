from gymnasium.envs.classic_control import CartPoleEnv


class CartPole_reward_1(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

    def step(self, action):
        state, reward, done, terminated, info = super().step(action)
        reward = self.modified_reward_function(state, reward)
        return state, reward, done, terminated, info

    def modified_reward_function(self, state, reward):
        cart_position, cart_velocity, pole_angle, pole_velocity = state

        return reward - abs(cart_position) * 0.5 - abs(pole_angle) * 1 - abs(cart_velocity) * 0.5
