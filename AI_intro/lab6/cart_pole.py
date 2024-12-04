import time
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random


class Agent:
    def __init__(self, action_space, num_of_states, eta=0.5, gamma=0.99, num=6):
        self.action_space = action_space
        self.num_of_states = num_of_states
        self.eta = eta
        self.gamma = gamma
        self.num = num
        self.q_table = np.random.uniform(0, 1, (num ** num_of_states, action_space.n))

    def get_action(self, state, episode):
        epsilon = 0.5 * 1 / (episode + 1)
        if random.random() < epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[state, :])

    def get_discrete_state(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation
        discrete = [
            np.digitize(cart_pos, bins=np.linspace(-2.4, 2.4, self.num)[1:-1]),
            np.digitize(cart_v, bins=np.linspace(-3.0, 3.0, self.num)[1:-1]),
            np.digitize(pole_angle, bins=np.linspace(-.21, .21, self.num)[1:-1]),
            np.digitize(pole_v, bins=np.linspace(-1.0, 1.0, self.num)[1:-1]),
        ]
        state = sum([self.num ** i * d for i, d in enumerate(discrete)])
        return state

    def q_learning(self, observation, action, reward, next_observation):
        observation_state = self.get_discrete_state(observation)
        next_observation_state = self.get_discrete_state(next_observation)
        max_q_value = np.max(self.q_table[next_observation_state, :])
        self.q_table[observation_state, action] += self.eta * (reward + self.gamma * max_q_value - self.q_table[observation_state, action])

    def get_reward(self, done, step, max_step):
        if done:
            if step < max_step:
                return -(max_step - step)
            return step // 10 + 1
        return 0


def display_frames_as_video(frames, n):
    plt.figure(figsize=(frames[0].shape[0] / 72.0, frames[0].shape[1] / 72.0), dpi=72)
    plt.axis('off')
    patch = plt.imshow(frames[0])
    def animate(index):
        patch.set_data(frames[index])
    animation = FuncAnimation(plt.gcf(), animate, frames=range(len(frames)), interval=50)
    animation.save(f'video/cart_pole{n}.gif', writer='ffmpeg')


env = gym.make('CartPole-v1', render_mode='rgb_array')
env.reset()
agent = Agent(env.action_space, env.observation_space.shape[0])
best_frames = []
max_step = 0
for i in range(1000):
    observation, _ = env.reset()
    done = False
    step = 0
    frames = []
    while not done:
        step += 1
        if step < 5000:
            frames.append(env.render())
        action = agent.get_action(agent.get_discrete_state(observation), i)
        observation_next, _, done, info, _ = env.step(action)
        reward = agent.get_reward(done, step, max_step)
        agent.q_learning(observation, action, reward, observation_next)
        observation = observation_next

    print(f"Episode {i + 1}: Steps = {step}")
    if max_step < step:
        best_frames = frames
        max_step = step
env.close()
display_frames_as_video(best_frames, max_step)
