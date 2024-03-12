#!/bin/bash
#SBATCH --ntasks-per-node=20
#SBATCH --time=48:00:00
#SBATCH --partition=paula
#SBATCH --gres=gpu:a30:2

module purge
source ~/.bashrc
conda activate mace_env

python3 scripts/run_train.py --name="tyrosine_energy_force_model_I6" --train_file="tyrosine_train.xyz" --valid_fraction=0.1 --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' --model="ScaleShiftMACE" --r_max=5.0 --batch_size=5 --correlation=3 --max_num_epochs=100 --ema --ema_decay=0.99 --default_dtype="float32" --seed=123 --device=cuda --hidden_irreps="128x0e + 128x1o + 128x2e" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=6 --forces_weight=100.0 --energy_weight=100.0 --error_table="TotalMAE" --n_energies=13
~
