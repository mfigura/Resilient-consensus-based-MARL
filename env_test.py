import os
import numpy as np
import gym
from gym import spaces
from environments.grid_world import Grid_World

'Quick test of the environment'

env = Grid_World(nrow=3,
                 ncol=3,
                 desired_state = np.random.randint(0,3,size=(3,2)),
                 n_agents=3,
                 scaling=False
                 )
env.reset()
for i in range(10):
    a_rand = np.random.randint(0,5,size=3)
    print(env.state)
    print(a_rand)
    env.step(a_rand)
    print(env.desired_state)
    print(env.reward)
