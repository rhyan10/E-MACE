#!/bin/bash
#SBATCH --ntasks-per-node=10
#SBATCH --time=48:00:00
#SBATCH --partition=paula
#SBATCH --gres=gpu:a30:1

module purge
source ~/.bashrc
conda activate mace_env

python3 scripts/run_train.py --name="CH2NH2_energy_model_dipole" --train_file="../ch2nh2/ch2nh2_final.xyz" --valid_fraction=0.1 --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' --model="EnergyDipolesMACE" --r_max=10.0 --batch_size=20 --correlation=3 --max_num_epochs=100 --ema --ema_decay=0.99 --default_dtype="float64" --seed=123 --device=cuda --hidden_irreps="128x0e + 128x1o + 128x2e" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=3 --forces_weight=100.0 --energy_weight=100.0 --dipole_weight=10.0 --error_table="DipoleMAE" --loss="energy_forces_dipole"
~