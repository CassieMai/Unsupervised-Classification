#!/bin/bash
#SBATCH -J debug2   # ======
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu1
#SBATCH -c 2
#SBATCH -N 1
#SBATCH -o out/train-scan.out    # eg.: MMDDii-train.out, MMDDi-test.out, debug.out  102901-train.out  052204-test
#SBATCH -w node2

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

nvidia-smi

# echo "run pest data prepare"
# python pestdata_prepare.py

# echo "run simclr"
# python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_pest.yml

echo "run scan"
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_pest.yml

# echo "run selflabel"
# python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_pest.yml
