#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=shire-1-1,shire-1-6,shire-1-10,shire-2-5,shire-2-9
#SBATCH --output=logs/datagen-%A-%a-%N.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=general

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

for max_num_examples in 20000; do
for success_threshold in 0.5; do

python ~/insta-dev/sft/create_sft_dataset.py \
    --max_num_examples $max_num_examples \
    --success_threshold $success_threshold

done
done
