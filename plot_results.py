import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

matplotlib.rcParams.update({'font.size': 100,'lines.linewidth':10})


def plot_returns():

    path = "./simulation_results/raw_data/"
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
                data = []
                for j in range(1,3):
                    current_data = pd.read_pickle(path4+"sim_data" + str(j) + ".pkl")
                    data.append(current_data[500:])
                sim_data[i][-1].append(pd.concat(data,ignore_index=True))

    '''Compute stats over simulations with different random seeds'''
    for i in range(len(sim_data)):
        sim_data_mean.append([])
        for j in range(len(sim_data[i])):
            sim_data_mean[i].append([])
            if len(sim_data[i][j]) != 0:
                sim_data_mean[i][j] = pd.concat(sim_data[i][j]).groupby(level=0).mean().rolling(200,min_periods=1).mean()

    '''Plot the team-average episode returns and the adversary's returns'''
    for i in range(0,len(sim_data_mean),2):
        for j in range(len(sim_data_mean[i])):
            t = np.arange(sim_data_mean[i][j].shape[0])
            print(t)
            fig,ax = plt.subplots(1,1,figsize=(48,27))
            ax.set_xlabel("Episode")
            ax.set_ylabel("Returns")

            #if 'coop' not in scenarios[i]:
                #ax.plot(t,sim_data_mean[i][j]["True_adv_returns"],label='True adv',color='r')
            ax.plot(t,sim_data_mean[i][j]["Estimated_team_returns"],label='Estimated',color='b')
            ax.plot(t,sim_data_mean[i][j]["True_team_returns"],label='True',color='g')
            ax.plot(t,sim_data_mean[i+1][j]["True_team_returns"],label='True (global)',color='r')
            ax.legend()
            plt.savefig('./simulation_results/figures/' + scenarios[i] + "_h" + str(j) + '.png')

if __name__ == '__main__':
    plot_returns()
