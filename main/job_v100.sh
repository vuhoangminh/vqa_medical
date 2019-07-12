#!/bin/bash
#SBATCH -A SNIC2018-3-406
#SBATCH -n 1
##SBATCH --exclusive
#SBATCH --time=07-00:00:00
#SBATCH --error=%J_error.out
#SBATCH --output=%J_output.out
#SBATCH --gres=gpu:v100:1

ml GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1

export PYTHONPATH=${PWD}:$PYTHONPATH

export KERAS_BACKEND="tensorflow"

export command="python main/loop_train_v100.py"
echo "$command"
srun $command

wait