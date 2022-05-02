import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

matplotlib.rcParams.update({'font.size': 50,'lines.linewidth':2})


def plot_rewards(sim_data,args):
    '''
    Plots the simulation results - individual accumulated episode rewards and team-average returns.
    '''
    ep_rewards = np.zeros((args['n_episodes'],args['n_agents']))
    estimated_ep_rewards = np.zeros((args['n_episodes'],args['n_agents']))
    ep_returns = np.zeros(args['n_episodes'])
    estimated_ep_returns = np.zeros(args['n_episodes'])
    for i,item in enumerate(sim_data):
        ep_rewards[i] = item["Episode_rewards"]
        ep_returns[i] = item["True_team_returns"]
        estimated_ep_returns[i] = item["Estimated_team_returns"]
    t = np.arange(len(sim_data))
    fig,ax = plt.subplots(1,args['n_agents']+1,sharey=False,figsize=(100,15))
    for i in range(args['n_agents']):
        ax[i].set_xlabel("Episode")
        ax[i].set_ylabel("Rewards")
        ax[i].plot(t,ep_rewards[:,i],label='True')
        ax[i].legend()
    ax[-1].set_xlabel("Episode")
    ax[-1].set_ylabel("Team-average returns")
    ax[-1].plot(t,ep_returns,label='True')
    ax[-1].plot(t,estimated_ep_returns,label='Est')
    ax[-1].legend()
    plt.savefig(args['summary_dir']+'sim_results.png')

def plot_returns():

    path = "./simulation_results_april22/raw_data/"
    scenarios = os.listdir(path)
    sim_data, sim_data_mean,sim_data_std = [],[],[]

    '''Retrieve simulation results from subdirectories'''
    for i,scenario in enumerate(scenarios):
        sim_data.append([])
        path2 = path + scenario + "/"
        H = os.listdir(path2)
        for h in H:
            sim_data[-1].append([])
            path3 = path2 + h + "/"
            random_seeds = os.listdir(path3)
            for rs in random_seeds:
                path4 = path3 + rs + "/"
                sim_data[i][-1].append(pd.read_pickle(path4+"sim_data.pkl"))

    '''Compute stats over simulations with different random seeds'''
    for i in range(len(sim_data)):
        sim_data_mean.append([])
        sim_data_std.append([])
        for j in range(len(sim_data[i])):
            sim_data_mean[i].append([])
            sim_data_std[i].append([])
            if len(sim_data[i][j]) != 0:
                sim_data_mean[i][j] = pd.concat(sim_data[i][j]).groupby(level=0).mean()
                sim_data_std[i][j] = pd.concat(sim_data[i][j]).groupby(level=0).std()

    '''Plot the 95% confidence intervals for the team-average episode returns and the adversary's returns'''
    for i in range(len(sim_data_mean)):
        for j in range(len(sim_data_mean[i])):
            t = np.arange(sim_data_mean[i][j].shape[0])
            fig,ax = plt.subplots(1,1,figsize=(30,15))
            ax.set_xlabel("Episode")
            ax.set_ylabel("Returns")

            if 'coop' not in scenarios[i]:
                ax.plot(t,sim_data_mean[i][j]["True_adv_returns"],label='True adv',color='r')
            ax.plot(t,sim_data_mean[i][j]["Estimated_team_returns"],label='Est coop',color='b')
            ax.plot(t,sim_data_mean[i][j]["True_team_returns"],label='True coop',color='g')
            ax.legend()
            plt.savefig('./simulation_results_april22/figures/' + scenarios[i] + "_h" + str(j) + '.png')

if __name__ == '__main__':
    plot_returns()
