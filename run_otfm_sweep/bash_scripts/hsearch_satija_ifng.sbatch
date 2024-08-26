#!/bin/bash

#SBATCH -o h-otfm-satija-ifng.out

#SBATCH -e h-otfm-satija-ifng.err

#SBATCH -J h-otfm-satija-ifng

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=80GB

#SBATCH -t 2-00:00

#SBATCH --nice=1

#source ${HOME}/.bashrc_new
source /lustre/groups/ml01/workspace/lea.zimmermann/software/miniconda3/etc/profile.d/conda.sh
conda activate ot

python /home/icb/lea.zimmermann/projects/cell_flow_perturbation/run_otfm_sweep/train_satija_ifng.py --multirun dataset=satija_ifng hparams_search=satija_ifng model=satija_ifng training=satija_ifng logger=satija_ifng launcher=slurm_icb
