import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import pandas as pd

'''
This file contains a function for training consensus AC agents in gym environments. It is designed for batch updates.
'''

def train_RPBCAC(env,agents,args):
    '''
    FUNCTION train_RBPCAC() - training a mixed cooperative and adversarial network of consensus AC agents including RPBCAC agents
    The agents apply actions sampled from the actor network and estimate online the team-average errors for the critic and team-average reward updates.
    At the end of a sequence of episodes, the agents update the actor, critic, and team-average reward parameters in batches. All participating agents
    transmit their critic and team reward parameters but only cooperative agents perform resilient consensus updates. The critic and team-average reward
    networks are used for the evaluation of the actor gradient. In addition to the critic and team reward updates, the adversarial agents separately
    update their local critic that is used in their actor updates.

    ARGUMENTS: gym environment
               list of resilient consensus AC agents
               user-defined parameters for the simulation
    '''
    paths = []
    n_agents, n_states, n_actions = env.n_agents, args['n_states'], env.n_actions
    gamma, eps = args['gamma'], args['eps']
    in_nodes = args['in_nodes']
    max_ep_len, n_episodes, n_ep_fixed = args['max_ep_len'], args['n_episodes'], args['n_ep_fixed']
    n_epochs = args['n_epochs']
    states = np.zeros((n_ep_fixed,max_ep_len+1,n_agents,n_states))
    actions = np.zeros((n_ep_fixed,max_ep_len,n_agents),dtype=int)
    rewards = np.zeros((n_ep_fixed,max_ep_len,n_agents))
    TR_errors_team = np.zeros((n_ep_fixed,max_ep_len,n_agents))
    critic_errors_team = np.zeros((n_ep_fixed,max_ep_len,n_agents))


    #writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    for t in range(n_episodes):
        #-----------------------------------------------------------------------
        '''TRAINING'''
        #-----------------------------------------------------------------------
        j,ep_rewards,ep_returns = 0,0,0
        est_returns, n_coop, mean_true_returns, mean_true_returns_adv = 0,0,0,0
        actor_loss,critic_loss,TR_loss = np.zeros(n_agents),np.zeros(n_agents),np.zeros(n_agents)
        i = t % n_ep_fixed
        env.reset()
        states[i,j], rewards[i,j], done, _ = env.get_data()

        'Evaluate expected retuns at the beginning of an episode'
        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                est_returns += agents[node].critic(states[i,j].reshape(1,-1))[0,0].numpy()
                n_coop += 1
        mean_est_returns = est_returns/n_coop

        'Simulate episode'
        while j < max_ep_len:
            for node in range(n_agents):
                  actions[i,j,node] = agents[node].get_action(states[i,j].reshape(1,-1),from_policy=True,mu=eps)
            env.step(actions[i,j])
            states[i,j+1], rewards[i,j], done, _ = env.get_data()
            ep_rewards += rewards[i,j]
            ep_returns += rewards[i,j]*(gamma**j)
            #------------------
            'END OF SIMULATION'
            #------------------

            if i == n_ep_fixed-1 and j == max_ep_len-1:
                #----------------------------------------------------------------------------------------
                '''BATCH CRITIC, TEAM REWARD, AND ACTOR UPDATES + RESILIENT PROJECTION-BASED CONSENSUS'''
                #----------------------------------------------------------------------------------------
                s = states[:,:-1].reshape(n_ep_fixed*max_ep_len,-1)
                ns = states[:,1:].reshape(n_ep_fixed*max_ep_len,-1)
                local_r = rewards.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                local_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                team_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents)
                critic_err = critic_errors_team.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                TR_err = TR_errors_team.reshape(n_ep_fixed*max_ep_len,n_agents,1)

                n = 0
                while n < n_epochs:

                    critic_weights,TR_weights = [],[]

                    #---------------------------------------------------
                    'BATCH LOCAL CRITIC AND TEAM-AVERAGE REWARD UPDATES'
                    #---------------------------------------------------
                    for node in range(n_agents):
                        if args['agent_label'][node] == 'Cooperative':
                            x, TR_loss[node] = agents[node].TR_update_local(s,team_a,local_r[:,node])
                            y, critic_loss[node] = agents[node].critic_update_local(s,ns,local_r[:,node])
                        elif args['agent_label'][node] == 'Greedy':
                            x = agents[node].TR_update_local(s,team_a,local_r[:,node])
                            y = agents[node].critic_update_local(s,ns,local_r[:,node])
                        elif args['agent_label'][node] == 'Malicious':
                            agents[node].critic_update_local(s,ns,local_r[:,node])
                            mean_coop_r = np.mean(np.delete(local_r,node,axis=1),axis=1)
                            x = agents[node].TR_update_compromised(s,team_a,-mean_coop_r)
                            y = agents[node].critic_update_compromised(s,ns,-mean_coop_r)
                        elif args['agent_label'][node] == 'Faulty':
                            x = agents[node].get_TR_weights()
                            y = agents[node].get_critic_weights()
                        elif args['agent_label'][node] == 'Byzantine':
                            agents[node].critic_update_local(s,ns,local_r[:,node])
                            x = agents[node].TR_attack(s,team_a,TR_weights)
                            y = agents[node].critic_attack(s,critic_weights)
                        TR_weights.append(x)
                        critic_weights.append(y)
                    #--------------------------------------------------------------------------------------------
                    'RESILIENT PROJECTION-BASED CONSENSUS UPDATES OF THE CRITIC AND TEAM-AVERAGE REWARD NETWORKS'
                    #--------------------------------------------------------------------------------------------
                    for node in (x for x in range(n_agents) if args['agent_label'][x] == 'Cooperative'):

                        'Aggregate parameters received from neighbors'
                        critic_weights_innodes = [critic_weights[i] for i in in_nodes[node]]
                        TR_weights_innodes = [TR_weights[i] for i in in_nodes[node]]

                        'Apply resilient projection-based consensus + element-wise trimming in hidden layers'
                        critic_est_trimmed = agents[node].resilient_consensus_critic(s,critic_weights_innodes)
                        TR_est_trimmed = agents[node].resilient_consensus_TR(s,team_a,TR_weights_innodes)
                        agents[node].resilient_consensus_hidden(critic_weights_innodes,TR_weights_innodes)

                        'Apply a stochastic update in the hidden layers with the mean estimated errors over neighbors'
                        x = agents[node].critic_update_team(s,ns,critic_est_trimmed)
                        y = agents[node].TR_update_team(s,team_a,TR_est_trimmed)

                    n += 1

                #----------------------------------------------
                'BATCH STOCHASTIC UPDATE OF THE ACTOR NETWORKS'
                #----------------------------------------------
                for node in range(n_agents):
                    if args['agent_label'][node] == 'Cooperative':
                        actor_loss[node] = agents[node].actor_update(s,ns,team_a,local_a[:,node])
                    else:
                        actor_loss[node] = agents[node].actor_update(s,ns,local_r[:,node],local_a[:,node])
            j += 1

        #------------------------------------
        '''SUMMARY OF THE TRAINING EPISODE'''
        #------------------------------------
        critic_mean_loss=np.mean(critic_loss)
        TR_mean_loss=np.mean(TR_loss)
        actor_mean_loss=np.mean(actor_loss)

        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                mean_true_returns += ep_returns[node]/n_coop
            else:
                mean_true_returns_adv += ep_returns[node]/(n_agents-n_coop)

        #with writer.as_default():
            #tf.summary.scalar("estimated episode team-average returns", mean_est_returns,step = t)
            #tf.summary.scalar("true episode team-average returns",mean_true_returns, step = t)
            #tf.summary.scalar("true episode team-average rewards",np.mean(ep_rewards), step = t)
            #writer.flush()

        print('| Episode: {} | Est. returns: {} | Returns: {} | Average critic loss: {} | Average TR loss: {} | Average actor loss: {} | Target reached: {} '.format(t,mean_est_returns,ep_returns,critic_loss,TR_loss,actor_loss,done))
        path = {
                "True_team_returns":mean_true_returns,
                "True_adv_returns":mean_true_returns_adv,
                "Estimated_team_returns":mean_est_returns
               }
        paths.append(path)

    sim_data = pd.DataFrame.from_dict(paths)
    return agents,sim_data

#----------------------------------------

def train_RTMCAC(env,agents,args):
    '''
    FUNCTION train_RBPCAC() - training a mixed cooperative and adversarial network of consensus AC agents including RCAC agents
    The agents apply actions sampled from the actor network and estimate online the team-average errors for the critic and team-average reward updates.
    At the end of a sequence of episodes, the agents update the actor, critic, and team-average reward parameters in batches. All participating agents
    transmit their critic and team reward parameters but only cooperative agents perform resilient consensus updates. The critic and team-average reward
    networks are used for the evaluation of the actor gradient. In addition to the critic and team reward updates, the adversarial agents separately
    update their local critic that is used in their actor updates.

    ARGUMENTS: gym environment
               list of resilient consensus AC agents
               user-defined parameters for the simulation
    '''
    paths = []
    n_agents, n_states, n_actions = env.n_agents, args['n_states'], env.n_actions
    gamma, eps = args['gamma'], args['eps']
    in_nodes = args['in_nodes']
    max_ep_len, n_episodes, n_ep_fixed = args['max_ep_len'], args['n_episodes'], args['n_ep_fixed']
    n_epochs = args['n_epochs']
    states = np.zeros((n_ep_fixed,max_ep_len+1,n_agents,n_states))
    actions = np.zeros((n_ep_fixed,max_ep_len,n_agents),dtype=int)
    rewards = np.zeros((n_ep_fixed,max_ep_len,n_agents))
    TR_errors_team = np.zeros((n_ep_fixed,max_ep_len,n_agents))
    critic_errors_team = np.zeros((n_ep_fixed,max_ep_len,n_agents))


    #writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    for t in range(n_episodes):
        #-----------------------------------------------------------------------
        '''TRAINING'''
        #-----------------------------------------------------------------------
        j,ep_rewards,ep_returns = 0,0,0
        est_returns, n_coop, mean_true_returns, mean_true_returns_adv = 0,0,0,0
        actor_loss,critic_loss,TR_loss = np.zeros(n_agents),np.zeros(n_agents),np.zeros(n_agents)
        i = t % n_ep_fixed
        env.reset()
        states[i,j], rewards[i,j], done, _ = env.get_data()

        'Evaluate estimated expected returns at the beginning of an episode'
        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                est_returns += agents[node].critic(states[i,j].reshape(1,-1))[0,0].numpy()
                n_coop += 1
        mean_est_returns = est_returns/n_coop

        'Simulate episode'
        while j < max_ep_len:
            for node in range(n_agents):
                  actions[i,j,node] = agents[node].get_action(states[i,j].reshape(1,-1),from_policy=True,mu=eps)
            env.step(actions[i,j])
            states[i,j+1], rewards[i,j], done, _ = env.get_data()
            ep_rewards += rewards[i,j]
            ep_returns += rewards[i,j]*(gamma**j)
            #------------------
            'END OF SIMULATION'
            #------------------

            if i == n_ep_fixed-1 and j == max_ep_len-1:

                #----------------------------------------------------------------------------------------
                '''BATCH CRITIC, TEAM REWARD, AND ACTOR UPDATES + RESILIENT TRIMMED-MEAN CONSENSUS'''
                #----------------------------------------------------------------------------------------
                s = states[:,:-1].reshape(n_ep_fixed*max_ep_len,-1)
                ns = states[:,1:].reshape(n_ep_fixed*max_ep_len,-1)
                local_r = rewards.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                local_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                team_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents)
                critic_err = critic_errors_team.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                TR_err = TR_errors_team.reshape(n_ep_fixed*max_ep_len,n_agents,1)

                n = 0
                while n < n_epochs:

                    critic_weights,TR_weights = [],[]

                    #---------------------------------------------------
                    'BATCH LOCAL CRITIC AND TEAM-AVERAGE REWARD UPDATES'
                    #---------------------------------------------------
                    for node in range(n_agents):
                        if args['agent_label'][node] == 'Cooperative':
                            x, TR_loss[node] = agents[node].TR_update_local(s,team_a,local_r[:,node])
                            y, critic_loss[node] = agents[node].critic_update_local(s,ns,local_r[:,node])
                        elif args['agent_label'][node] == 'Greedy':
                            x = agents[node].TR_update_local(s,team_a,local_r[:,node])
                            y = agents[node].critic_update_local(s,ns,local_r[:,node])
                        elif args['agent_label'][node] == 'Malicious':
                            agents[node].critic_update_local(s,ns,local_r[:,node])
                            mean_coop_r = np.mean(np.delete(local_r,node,axis=1),axis=1)
                            x = agents[node].TR_update_compromised(s,team_a,-mean_coop_r)
                            y = agents[node].critic_update_compromised(s,ns,-mean_coop_r)
                        elif args['agent_label'][node] == 'Faulty':
                            x = agents[node].get_TR_weights()
                            y = agents[node].get_critic_weights()

                        TR_weights.append(x)
                        critic_weights.append(y)

                    #--------------------------------------------------------------------------------------------
                    'RESILIENT TRIMMED-MEAN CONSENSUS UPDATES OF THE CRITIC AND TEAM-AVERAGE REWARD NETWORKS'
                    #--------------------------------------------------------------------------------------------
                    for node in (x for x in range(n_agents) if args['agent_label'][x] == 'Cooperative'):

                        'Aggregate parameters received from neighbors'
                        critic_weights_innodes = [critic_weights[i] for i in in_nodes[node]]
                        TR_weights_innodes = [TR_weights[i] for i in in_nodes[node]]

                        'Apply element-wise trimming over critic and team-average reward function parameters'
                        agents[node].resilient_consensus(critic_weights_innodes,TR_weights_innodes)

                    n += 1
                #----------------------------------------------
                'BATCH STOCHASTIC UPDATE OF THE ACTOR NETWORKS'
                #----------------------------------------------
                for node in range(n_agents):
                    if args['agent_label'][node] == 'Cooperative':
                        actor_loss[node] = agents[node].actor_update(s,ns,team_a,local_a[:,node])
                    else:
                        actor_loss[node] = agents[node].actor_update(s,ns,local_r[:,node],local_a[:,node])
            j += 1

        #-----------------------------------------------------------------------
        '''SUMMARY OF THE TRAINING EPISODE'''
        #-----------------------------------------------------------------------
        critic_mean_loss=np.mean(critic_loss)
        TR_mean_loss=np.mean(TR_loss)
        actor_mean_loss=np.mean(actor_loss)

        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                mean_true_returns += ep_returns[node]/n_coop
            else:
                mean_true_returns_adv += ep_returns[node]/(n_agents-n_coop)

        #with writer.as_default():
            #tf.summary.scalar("estimated episode team-average returns", mean_est_returns,step = t)
            #tf.summary.scalar("true episode team-average returns",mean_true_returns, step = t)
            #tf.summary.scalar("true episode team-average rewards",np.mean(ep_rewards), step = t)
            #writer.flush()

        print('| Episode: {} | Est. returns: {} | Returns: {} | Average critic loss: {} | Average TR loss: {} | Average actor loss: {} | Target reached: {} '.format(t,mean_est_returns,ep_returns,critic_loss,TR_loss,actor_loss,done))
        path = {
                #"Episode_rewards":np.array(ep_rewards),
                "True_team_returns":mean_true_returns,
                "True_adv_returns":mean_true_returns_adv,
                "Estimated_team_returns":mean_est_returns
               }
        paths.append(path)

    sim_data = pd.DataFrame.from_dict(paths)
    return agents,sim_data
