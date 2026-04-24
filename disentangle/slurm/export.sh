#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu
#SBATCH --time=3:10:00
#SBATCH --account=shrikann_35
#SBATCH --mem=40G
#SBATCH --output=/project2/shrikann_35/nmehlman/logs/private_codecs/slurm/%j_output.log
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a40:1
#SBATCH --chdir /home1/nmehlman/private_codecs/private_codecs/disentangle
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmehlman@usc.edu

module purge
source /project2/shrikann_35/nmehlman/conda/etc/profile.d/conda.sh
conda activate priv-codec

CONFIG="$1"

srun python misc/export_data.py --config "$CONFIG" 
