#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=56GB
#SBATCH --job-name=EXP_GEN
#SBATCH --account=pr_95_tandon_priority
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr6177@nyu.edu
#SBATCH --output=slurm_%j.out

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ../../attack-analysis/penv;
export PATH=../../attack-analysis/penv/bin:$PATH;

python run.py & python experiments/baseline.py
