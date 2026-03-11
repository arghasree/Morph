#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2 


# module load python/3.10 cuda cudnn 
# module load python/3.11 cuda/12.9 cudnn/9.13.1.26

module load gcc opencv
module load mujoco python

# virtualenv --no-download /home/arghasre/scratch/CMPUT\ 605/cmput605/bin/activate

# python3 -m venv /scratch/arghasre/morpho-env
# virtualenv --no-download /scratch/arghasre/morpho-env
source /scratch/arghasre/morpho-env/bin/activate

# pip3 install --no-index -r requirement.txt
# pip3 install --no-index gymnasium

# virtualenv --no-download "/home/arghasre/scratch/CMPUT 605/cmput605"
# source /home/arghasre/scratch/CMPUT\ 605/cmput605/bin/activate
echo $VIRTUAL_ENV 
# pip install --no-index PyOpenGL PyOpenGL_accelerate      
which python3
# pip show PyOpenGL

nvidia-smi

# pip3 install -r requirements.txt
# python -m pip install --no-cache-dir -r requirements.txt

# pip uninstall mujoco-py -y && pip install --no-cache-dir mujoco-py==2.1.2.14

# python3 main.py
# sh scripts/Run_BodyGen.sh
python3 controller.py 
# --PPO_n_steps 2048 --PPO_n_epochs 10 --PPO_total_steps 10000