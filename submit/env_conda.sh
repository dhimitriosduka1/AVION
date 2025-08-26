#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.err

#SBATCH --job-name avion_env

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=00:59:59

module purge
module load gcc/11
module load cuda/11.8-nvhpcsdk
module load anaconda/3/2021.11

# Remove env if it already exists
conda env remove --name avion -y

conda create --name avion python=3.10 -y
conda activate avion
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja==1.11.1
CUDA_HOME=$CUDA_HOME pip install -r requirements.txt

conda install -c conda-forge decord

# conda install -c conda-forge ffmpeg

# cd /u/dduka/work/projects/AVION
# cd third_party/decord/

# rm -rf build

# mkdir build && cd build
# cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_INCLUDE_DIR=$CONDA_PREFIX/include -DFFMPEG_LIBRARIES=$CONDA_PREFIX/lib
# make

# cd ../python
# python3 setup.py install --user

# PYTHONPATH=$PYTHONPATH:$PWD

# python -c "import decord; print(decord.__path__)"

# module purge
# module load anaconda/3/2021.11
# module load cuda/11.8-nvhpcsdk

# cd /u/dduka/work/projects/AVION

# conda create --name avion python=3.10 -y
# conda activate avion
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install ninja==1.11.1
# CUDA_HOME=$CUDA_HOME pip install -r requirements.txt

# # Install ffmpeg
# conda install -c conda-forge "ffmpeg<5"

# cd /u/dduka/work/projects/AVION/third_party/decord/
# rm -rf build
# mkdir build && cd build

# CONDA_PREFIX=/u/dduka/conda-envs/avion
# cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_INCLUDE_DIR=$CONDA_PREFIX/include -DFFMPEG_LIBRARIES=$CONDA_PREFIX/lib

# make

# cd ../python
# python3 setup.py install --user

# PYTHONPATH=$PYTHONPATH:$PWD

# python -c "import decord; print(decord.__path__)"