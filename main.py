import os
import numpy as np
import gym
import argparse
import pickle
import pandas as pd
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
from environments.grid_world import Grid_World
from agents.resilient_CAC_agents import RPBCAC_agent, RTMCAC_agent
from agents.adversarial_CAC_agents import Faulty_CAC_agent, Greedy_CAC_agent, Malicious_CAC_agent, Byzantine_CAC_agent
import training.train_agents as training

'''
Cooperative navigation problem with resilient consensus and adversarial actor-critic agents
- This is a main file, where the user selects learning hyperparameters, environment parameters,
  and neural network architecture for the actor, critic, and team reward estimates.
- The script triggers a training process whose results are passed to folder Simulation_results.
'''

if __name__ == '__main__':

    '''USER-DEFINED PARAMETERS'''
    parser = argparse.ArgumentParser(description='Provide parameters for training consensus AC agents')
    parser.add_argument('--n_agents',help='total number of agents',type=int,default=5)
    parser.add_argument('--agent_label', help='classification of each agent (Cooperative,Malicious,Faulty,Greedy)',type=str, default=['Cooperative','Cooperative','Cooperative','Cooperative','Malicious'])
    parser.add_argument('--n_actions',help='size of action space of each agent',type=int,default=5)
    parser.add_argument('--n_states',help='state dimension of each agent',type=int,default=2)
    parser.add_argument('--n_episodes', help='number of episodes', type=int, default=10000)
    parser.add_argument('--max_ep_len', help='Number of steps per episode', type=int, default=20)
    parser.add_argument('--slow_lr', help='actor network learning rate',type=float, default=0.002)
    parser.add_argument('--fast_lr', help='critic network learning rate',type=float, default=0.01)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.9)
    parser.add_argument('--H', help='max number of adversaries in the local neighborhood', type=int, default=1)
    parser.add_argument('--eps', help='exploration noise',type=float,default=0.05)
    parser.add_argument('--n_ep_fixed',help='number of episodes under a fixed policy',type=int,default=100)
    parser.add_argument('--n_epochs',help='number of gradient steps in the critic and team reward updates',type=int,default=20)
    parser.add_argument('--in_nodes',help='specify a list of neighbors that transmit values to each agent (include the index of the agent as the first element)',type=int,default=[[0,1,2,3],[1,2,3,4],[2,3,4,0],[3,4,0,1],[4,0,1,2]])
    parser.add_argument('--randomize_state',help='Set to True if the agents start at random initial state in every episode',type=bool,default=True)
    parser.add_argument('--scaling', help='Normalize states for training?', type = bool, default=True)
    parser.add_argument('--resilient_method', help='Choose between trimmed mean and projection-based consensus', default='projection-based')
    parser.add_argument('--summary_dir',help='Create a directory to save simulation results', default='./simulation_results/')
    parser.add_argument('--random_seed',help='Set random seed for the random number generator',type=int,default=20)
    parser.add_argument('--desired_state',help='desired state of the agents',type=int,default=np.random.randint(0,6,size=(4,2)))
    parser.add_argument('--initial_state',help='initial state of the agents',type=int,default=np.random.randint(0,6,size=(4,2)))
    args = vars(parser.parse_args())
    np.random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])
    args['desired_state'] = np.random.randint(0,6,size=(args['n_agents'],args['n_states']))
    args['initial_state'] = np.random.randint(0,6,size=(args['n_agents'],args['n_states']))
    #folder_path = os.path.join(os.getcwd(),'simulation_results/scenarios/' + args['resilient_method'] + '/' + args['agent_label'][-1] + '_h' + str(args['H']) + '/seed=' + str(args['random_seed']) + '/')
    #if not os.path.isdir(folder_path):
    #    os.makedirs(folder_path)

    #----------------------------------------------------------------------------------------------------------------------------------------
    '''NEURAL NETWORK ARCHITECTURE'''
    agents = []

    critic_template = keras.Sequential([
                                keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1),input_shape=(args['n_agents']*args['n_states'],)),
                                keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                #keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                keras.layers.Dense(1)
                              ])
    team_reward_template = keras.Sequential([
                                keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1),input_shape=(args['n_agents']*args['n_states']+args['n_agents'],)),
                                keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                #keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                keras.layers.Dense(1)
                              ])

    for node in range(args['n_agents']):

        actor = keras.Sequential([
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1),input_shape=(args['n_agents']*args['n_states'],)),
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    #keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                    keras.layers.Dense(args['n_actions'], activation='softmax')
                                  ])

        critic = keras.Sequential([
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1),input_shape=(args['n_agents']*args['n_states'],)),
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    #keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                    keras.layers.Dense(1)
                                  ])
        critic2 = keras.Sequential([
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1),input_shape=(args['n_agents']*args['n_states'],)),
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    #keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                    keras.layers.Dense(1)
                                  ])

        team_reward = keras.Sequential([
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1),input_shape=(args['n_agents']*args['n_states']+args['n_agents'],)),
                                    keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    #keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                    keras.layers.Dense(1)
                                  ])

        critic.set_weights(critic_template.get_weights())
        critic2.set_weights(critic_template.get_weights())
        team_reward.set_weights(team_reward_template.get_weights())

        if args['agent_label'][node] == 'Malicious':        #create a malicious agent
            print("This is a malicious agent")
            agents.append(Malicious_CAC_agent(actor,critic,critic2,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma']))

        elif args['agent_label'][node] == 'Faulty':         #create a faulty agent
            print("This is a faulty agent")
            agents.append(Faulty_CAC_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],gamma = args['gamma']))

        elif args['agent_label'][node] == 'Greedy':         #create a greedy agent
            print("This is a greedy agent")
            agents.append(Greedy_CAC_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma']))

        elif args['agent_label'][node] == 'Byzantine':         #create a greedy agent
            print("This is a Byzantine agent")
            agents.append(Byzantine_CAC_agent(actor,critic,critic2,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma']))

        elif args['agent_label'][node] == 'Cooperative':    #create a cooperative agent
            if args['resilient_method'] == 'projection-based':
                print("This is an RPBCAC agent")                                          #create a cooperative agent
                agents.append(RPBCAC_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma'],H = args['H']))
            elif args['resilient_method'] == 'trimmed-mean':
                print("This is an RTMCAC agent")
                agents.append(RTMCAC_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma'],H = args['H']))

    print(args)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    '''TRAIN AGENTS'''
    env = Grid_World(nrow=6,
                     ncol=6,
                     n_agents=args['n_agents'],
                     desired_state=args['desired_state'],
                     initial_state=args['initial_state'],
                     randomize_state=args['randomize_state'],
                     scaling=args['scaling']
                     )
    if args['resilient_method'] == 'projection-based':
        trained_agents,sim_data = training.train_RPBCAC(env,agents,args)
    else:
        trained_agents,sim_data = training.train_RTMCAC(env,agents,args)
    #----------------------------------------------------------------------------------------------------
    sim_data.to_pickle("sim_data.pkl")
