#!/bin/bash
#SBATCH --ntasks-per-node=10
#SBATCH --time=48:00:00
#SBATCH --partition=paula
#SBATCH --gres=gpu:a30:1

module purge
source ~/.bashrc
conda activate mace_env

python3 scripts/run_train.py --name="CH2NH2_energy_model_norefs" --train_file="ch2nh2/ch2nh2_esps_cheb.xyz" --valid_fraction=0.1 --E0s='{1:0.0, 6:0.0, 7:0.0, 8:0.0}' --model="ScaleShiftMACE" --r_max=10.0 --batch_size=5 --correlation=3 --max_num_epochs=500 --ema --lr=0.01 --ema_decay=0.99 --default_dtype="float64" --seed=123 --device=cuda --hidden_irreps="128x0e + 128x1o + 128x2e" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=4 --forces_weight=0.0 --energy_weight=100.0 --error_table="TotalMAE" --restart_latest 
~
