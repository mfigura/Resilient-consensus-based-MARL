import numpy as np
import gym
from gym import spaces

class Grid_World(gym.Env):
    """
    Multi-agent grid-world: cooperative navigation 2
    This is a grid-world environment designed for the cooperative navigation problem. Each agent seeks to navigate to the desired position. The agent chooses one of five admissible actions
    (stay,left,right,down,up) and makes a transition only if the adjacent cell is not occupied. It receives a reward equal to the L1 distance between the visited cell and the target.
    ARGUMENTS:  nrow, ncol: grid world dimensions
                n_agents: number of agents
                desired_state: desired position of each agent
                initial_state: initial position of each agent
                randomize_state: True if the agents' initial position is randomized at the beginning of each episode
                scaling: determines if the states are scaled
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, nrow = 5, ncol=5, n_agents = 1, desired_state = None,initial_state = None,randomize_state = True,scaling = False):
        self.nrow = nrow
        self.ncol = ncol
        self.n_agents = n_agents
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.randomize_state = randomize_state
        self.n_states = 2
        self.actions_dict = {0:np.array([0,0]), 1:np.array([-1,0]), 2:np.array([1,0]), 3:np.array([0,-1]), 4:np.array([0,1])}
        self.reset()

        if scaling:
            x,y=np.arange(nrow),np.arange(ncol)
            self.mean_state=np.array([np.mean(x),np.mean(y)])
            self.std_state=np.array([np.std(x),np.std(y)])
        else:
            self.mean_state,self.std_state=0,1

    def reset(self):
        '''Resets the environment'''
        if self.randomize_state:
            self.state = np.random.randint([0,0],[self.nrow,self.ncol],size=(self.n_agents,self.n_states))
        else:
            self.state = np.array(self.initial_state)
        self.reward = np.zeros(self.n_agents)

        return self.state

    def step(self, action):
        '''
        Makes a transition to a new state and evaluates all rewards
        Arguments: global action
        '''
        for node in range(self.n_agents):
            move = self.actions_dict[action[node]]
            dist_to_goal = np.sum(abs(self.state[node]-self.desired_state[node]))
            self.state[node] = np.clip(self.state[node] + move,0,self.nrow - 1)
            dist_to_agents = np.min(np.sum(abs(self.state-self.state[node]),axis=1))
            dist_to_goal_next = np.sum(abs(self.state[node]-self.desired_state[node]))

            if dist_to_agents > 0: #agent moves to a new cell
                self.reward[node] = - dist_to_goal_next
            elif dist_to_goal == 0 and action[node] == 0:
                self.reward[node] = 0
            else:
                self.reward[node] = - dist_to_goal - 1

    def get_data(self):
        '''
        Returns scaled reward and state, and flags if the agents have reached the target
        '''
        state_scaled = (self.state - self.mean_state) / self.std_state
        reward_scaled = self.reward / 5
        return state_scaled, reward_scaled

    def close(self):
        pass
