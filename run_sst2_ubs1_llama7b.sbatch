#!/usr/bin/env bash
#SBATCH --nodes=1        
#SBATCH -t 1-12:00                                # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1        # Request gpu resources
#SBATCH -p seas_gpu                               # Partition: gpu
#SBATCH -v                                        # Make out more verbose 
#SBATCH --mem=80000                               # RAM requirement
#SBATCH -o myoutput_%j.out                        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err                        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=email address com             # info when job finishes

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/11.3.1-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

source activate p311

python run_llama.py --dataset_name "sst2" --dataset_size 25000 --config config_run_ubs1_llama7b.json
