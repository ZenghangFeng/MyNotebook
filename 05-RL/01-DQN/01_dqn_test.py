import gym
import numpy as np


""""""
# env = gym.make("CartPole-v0", render_mode="human")
env = gym.make("Pendulum-v1", render_mode="human")

env.reset()
# acts = env.action_space.n

for _ in range(1000):
    env.render()
    # act = np.random.choice(acts, 1)[0]
    env.step(env.action_space.sample()) # take a random action
env.close()


"""
class Test:
    def __init__(self, s1):
        self.s1 = s1
        self.s2 = s1

    def update_s2(self):
        self.s2 = self.s1

t1 = Test(1)
print(t1.s1, t1.s2)
for i in range(10):
    t1.s1 += 1
    if i == 9:
        t1.update_s2()
    print(t1.s1, t1.s2)
"""