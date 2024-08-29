#!/bin/bash

#SBATCH -o otfm_one_satija_ifng.out

#SBATCH -e otfm_one_satija_ifng.err

#SBATCH -J otfm_one_satija_ifng

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-00:20:00

#SBATCH --nice=1

#source ${HOME}/.bashrc_new
source /lustre/groups/ml01/workspace/lea.zimmermann/software/miniconda3/etc/profile.d/conda.sh
conda activate ot

python /home/icb/lea.zimmermann/projects/cell_flow_perturbation/run_otfm_sweep/train_satija_ifng.py dataset=satija_ifng logger=satija_ifng training=satija_ifng launcher=slurm_icb model=satija_ifng
