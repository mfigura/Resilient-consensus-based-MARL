#!/bin/bash
#$ -M mfigura@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q long
#$ -N Job=greedy4k,0.002,1,100
#$ -o info
module load conda
conda activate MARL_env
/afs/crc.nd.edu/user/m/mfigura/.conda/envs/MARL_env/bin/python /afs/crc.nd.edu/user/m/mfigura/Private/Cooperative_MARL/Resilient_consensus_AC/main.py --H=1 --slow_lr=0.002 --random_seed=100 > out.txt
