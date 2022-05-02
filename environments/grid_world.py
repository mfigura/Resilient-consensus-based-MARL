import numpy as np
import gym
from gym import spaces

class Grid_World(gym.Env):
    """
    Multi-agent grid-world: cooperative navigation
    This is a grid-world environment designed for the cooperative navigation problem.
    Each agent seeks to navigate to the desired position without colliding with other agents.
    The rewards are individually awarded for reaching the target and collisions.
    1) The reward for approaching the target is given as the negative Manhattan distance
       between the target and the agent at the new state.
    2) The penalty for collision is 0.5 times the reward for approaching the target.
    ARGUMENTS:  nrow, ncol: grid world dimensions
                n_agents: number of agents
                desired_state: desired position of each agent
                initial_state: initial position of each agent
                randomize_state: True if the agents' initial position is randomized at the beginning of each episode
                scaling: determines if the states are scaled
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, nrow = 5, ncol=5, n_agents = 1, desired_state = None,initial_state = None,randomize_state = False,scaling = False):
        self.nrow = nrow
        self.ncol = ncol
        self.n_agents = n_agents
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.randomize_state = randomize_state
        self.total_states = self.nrow * self.ncol
        self.n_states = 2
        self.n_actions = n_agents
        self.actions = {0:'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP', 4:'STAY'}
        self.reward=np.zeros(self.n_agents)
        self.observation_space = gym.spaces.MultiDiscrete([self.total_states for _ in range(self.n_agents)])
        self.action_space = gym.spaces.MultiDiscrete([self.n_actions for _ in range(self.n_agents)])
        self.reset()
        self.reward, self.done = np.full_like(self.n_agents, 0.0), False

        if scaling:
            x,y=np.arange(nrow),np.arange(ncol)
            self.mean_state=np.array([np.mean(x),np.mean(y)])
            self.std_state=np.array([np.std(x),np.std(y)])
        else:
            self.mean_state,self.std_state=0,1

    def _state_transition(self, local_state, local_action):
        '''
        Computes a new local state wrt to the current state and action
        Arguments: local state and local action
        Returns: new local state
        local action:  0 - LEFT
                       1 - DOWN
                       2 - RIGHT
                       3 - UP
                       4 - STAY
        '''
        row = local_state[0]
        col = local_state[1]
        if local_action == 0:
            col = max(col - 1, 0)
        elif local_action == 1:
            row = max(row - 1, 0)
        elif local_action == 2:
            col = min(col + 1, self.ncol - 1)
        elif local_action == 3:
            row = min(row + 1, self.nrow - 1)
        return np.array([row,col])

    def reset(self):
        '''Resets the environment'''
        if self.randomize_state:
            self.state = np.random.randint([0,0],[self.nrow,self.ncol],size=self.initial_state.shape)
        else:
            self.state = np.array(self.initial_state)
        self.reward, self.done = np.zeros(self.n_agents), False

        return self.state

    def step(self, global_action):
        '''
        Makes a transition to a new state and evaluates all rewards
        Arguments: global action
        '''
        ns = np.zeros((self.n_agents,self.n_states))
        'State transition'
        for node,s,a in zip(range(self.n_agents),self.state, global_action):
            ns[node] = self._state_transition(s,a)                                                     #State transition
        'Rewards'
        for node in range(self.n_agents):
            dist_agents = np.sort(np.sum(abs(ns-ns[node]),axis=1)).ravel()
            dist_goal = np.sum(abs(self.state[node]-self.desired_state[node]))
            dist_goal_next = np.sum(abs(ns[node]-self.desired_state[node]))
            r_approach = dist_goal - dist_goal_next                                                 #Reward +1 for approaching, -1 for moving away'
            r_wall = - int(dist_goal == dist_goal_next and global_action[node] != 4)                #Reward -1 for hitting the wall
            r_target = int(dist_goal_next == 0)                                                     #Reward +1 for staying at the target
            r_safe = - 1 / (3 ** dist_agents[1]) if self.n_agents > 1 else 0                        #Reward -1 for collision or -1/3 for being one cell away from another agent, etc.
            self.reward[node] = r_approach + r_wall + r_target + r_safe
        self.state = ns
        self.done = np.array_equal(self.state, self.desired_state)

    def get_data(self):
        '''
        Returns scaled reward and state, and flags if the agents have reached the target
        '''
        state_scaled = (self.state - self.mean_state) / self.std_state
        reward_scaled = self.reward
        return state_scaled, reward_scaled, self.done, {}

    def close(self):
        pass
