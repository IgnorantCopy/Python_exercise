import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

cart_v_bin = np.linspace(-2., 2., 6)
cart_vs = []
pole_v_bin = np.linspace(-2., 2., 6)
pole_vs = []

env = gym.make('CartPole-v1')
for i in range(1000):
    env.reset()
    done = False
    step = 0
    print(f"Episode {i + 1}:")
    while not done:
        step += 1
        action = env.action_space.sample()
        observation, reward, done, info, _ = env.step(action)
        cart_vs.append(observation[1])
        pole_vs.append(observation[3])

plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(cart_vs, bins=cart_v_bin)
plt.xlabel("Cart velocity")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(pole_vs, bins=pole_v_bin)
plt.xlabel("Pole angle")
plt.ylabel("Frequency")
plt.savefig("./sample.png")
# plt.show()
