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

def train_batch(env,agents,args):
    '''
    FUNCTION train_batch() - training a mixed cooperative and adversarial network
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


    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    for t in range(n_episodes):
        #-----------------------------------------------------------------------
        '''BEGINNING OF TRAINING EPISODE'''
        j,ep_rewards,ep_returns = 0,0,0
        est_returns, n_coop, mean_true_returns, mean_true_returns_adv = 0,0,0,0
        actor_loss,critic_loss,TR_loss = np.zeros(n_agents),np.zeros(n_agents),np.zeros(n_agents)
        i = t % n_ep_fixed
        env.reset()
        states[i,j], rewards[i,j], done, _ = env.get_data()
        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                est_returns += agents[node].critic(states[i,j].reshape(1,-1))[0,0].numpy()
                n_coop += 1
        mean_est_returns = est_returns/n_coop

        while j < max_ep_len:
            '''EPISODE SIMULATION'''
            for node in range(n_agents):
                  actions[i,j,node] = agents[node].get_action(states[i,j].reshape(1,-1),from_policy=True,mu=eps)
            env.step(actions[i,j])
            states[i,j+1], rewards[i,j], done, _ = env.get_data()
            ep_rewards += rewards[i,j]
            ep_returns += rewards[i,j]*(gamma**j)
            #-------------------------------------------------------------------
            '''BATCH CRITIC, TEAM REWARD, AND ACTOR UPDATES + RESILIENT CONSENSUS UPDATES OF HIDDEN LAYERS'''
            if i == n_ep_fixed-1 and j == max_ep_len-1:
                s = states[:,:-1].reshape(n_ep_fixed*max_ep_len,-1)
                ns = states[:,1:].reshape(n_ep_fixed*max_ep_len,-1)
                local_r = rewards.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                local_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                team_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents)
                critic_err = critic_errors_team.reshape(n_ep_fixed*max_ep_len,n_agents,1)
                TR_err = TR_errors_team.reshape(n_ep_fixed*max_ep_len,n_agents,1)

                'ITERATING OVER EPOCHS'
                n = 0
                while n < n_epochs:
                    #-------------------------------------------------------------------------------------------------------------------
                    'ONLINE STOCHASTIC UPDATES FOR ESTIMATION OF THE TEAM-AVERAGE ERROR'
                    for k in range(n_ep_fixed*max_ep_len):

                        critic_out,TR_out = [],[]

                        for node in range(n_agents):
                            if args['agent_label'][node] == 'Cooperative':
                                x = agents[node].TR_update_local(s[k:k+1],team_a[k:k+1],local_r[k:k+1,node])
                                y = agents[node].critic_update_local(s[k:k+1],ns[k:k+1],local_r[k:k+1,node])
                            elif args['agent_label'][node] == 'Greedy':
                                _ , x , _ = agents[node].TR_update_local(s[k:k+1],team_a[k:k+1],local_r[k:k+1,node],reset=True)
                                _ , y , _ = agents[node].critic_update_local(s[k:k+1],ns[k:k+1],local_r[k:k+1,node],reset=True)
                            elif args['agent_label'][node] == 'Faulty':
                                _ , x , _ = agents[node].get_fixed_critic()
                                _ , y , _ = agents[node].get_fixed_TR()
                            elif args['agent_label'][node] == 'Malicious':
                                coop_r = np.delete(local_r[k:k+1],node,axis=1)
                                mean_coop_r = np.mean(coop_r,axis=1)
                                _ , x , _ = agents[node].TR_update_compromised(s[k:k+1],team_a[k:k+1],-mean_coop_r,reset=True)
                                _ , y , _ = agents[node].critic_update_compromised(s[k:k+1],ns[k:k+1],-mean_coop_r,reset=True)
                            TR_out.append(x)
                            critic_out.append(y)

                        for node in range(n_agents):
                            if args['agent_label'][node] == 'Cooperative':
                                critic_out_innodes = [critic_out[i] for i in in_nodes[node]]
                                TR_out_innodes = [TR_out[i] for i in in_nodes[node]]
                                critic_err[k,node],TR_err[k,node] = agents[node].resilient_consensus(critic_out_innodes,TR_out_innodes)
                    #--------------------------------------------------------------------------------------------------------------------
                    'BATCH STOCHASTIC UPDATES OF THE CRITIC AND TEAM-AVERAGE REWARD NETWORKS'

                    critic_hidden,TR_hidden = [],[]

                    for node in range(n_agents):
                        if args['agent_label'][node] == 'Cooperative':
                            x, critic_loss[node] = agents[node].critic_update_team(s,ns,critic_err[:,node])
                            y, TR_loss[node] = agents[node].TR_update_team(s,team_a,TR_err[:,node])
                        elif args['agent_label'][node] == 'Greedy':
                            x, _ , critic_loss[node] = agents[node].critic_update_local(s,ns,local_r[:,node],reset=False)
                            y, _ , TR_loss[node] = agents[node].TR_update_local(s,team_a,local_r[:,node],reset=False)
                        elif args['agent_label'][node] == 'Faulty':
                            x , _ , critic_loss[node] = agents[node].get_fixed_critic()
                            y , _ , TR_loss[node] = agents[node].get_fixed_TR()
                        elif args['agent_label'][node] == 'Malicious':
                            coop_r = np.delete(local_r,node,axis=1)
                            mean_coop_r = np.mean(coop_r,axis=1)
                            agents[node].critic_update(s,ns,local_r[:,node])
                            x , _ , critic_loss[node] = agents[node].critic_update_compromised(s,ns,-mean_coop_r,reset=False)
                            y , _ , TR_loss[node] = agents[node].TR_update_compromised(s,team_a,-mean_coop_r,reset=False)
                        critic_hidden.append(x)
                        TR_hidden.append(y)

                    print(TR_loss,critic_loss)

                    'AVERAGING OF THE HIDDEN LAYER PARAMETERS USING THE RESILIENT CONSENSUS METHOD'
                    for node in range(n_agents):
                        if args['agent_label'][node] == 'Cooperative':
                            critic_hidden_innodes = [critic_hidden[i] for i in in_nodes[node]]
                            agents[node].resilient_consensus_hidden(critic_hidden_innodes,agents[node].critic.trainable_variables[:-2])
                            TR_hidden_innodes = [TR_hidden[i] for i in in_nodes[node]]
                            agents[node].resilient_consensus_hidden(TR_hidden_innodes,agents[node].TR.trainable_variables[:-2])
                    #--------------------------------------------------------------------------------------------------------------------
                    n += 1

                'BATCH STOCHASTIC UPDATE OF THE ACTOR NETWORKS'
                for node in range(n_agents):
                    if args['agent_label'][node] == 'Cooperative':
                        actor_loss[node] = agents[node].actor_update(s,ns,team_a,local_a[:,node])
                    else:
                        actor_loss[node] = agents[node].actor_update(s,ns,local_r[:,node],local_a[:,node])
            j += 1
        #-----------------------------------------------------------------------
        '''SUMMARY OF THE TRAINING EPISODE'''
        critic_mean_loss=np.mean(critic_loss)
        TR_mean_loss=np.mean(TR_loss)
        actor_mean_loss=np.mean(actor_loss)

        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                mean_true_returns += ep_returns[node]/n_coop
            else:
                mean_true_returns_adv += ep_returns[node]/(n_agents-n_coop)

        with writer.as_default():
            tf.summary.scalar("estimated episode team-average returns", mean_est_returns,step = t)
            tf.summary.scalar("true episode team-average returns",mean_true_returns, step = t)
            tf.summary.scalar("true episode team-average rewards",np.mean(ep_rewards), step = t)
            writer.flush()
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
