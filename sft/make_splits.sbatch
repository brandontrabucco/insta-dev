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

# Create SFT data based on qwen3-235b judge

judgments_name="judgments-qwen-235b"

for max_num_examples in 2000 5000 10000 20000; do
for success_threshold in 0.5; do

python ~/insta-dev/sft/create_sft_dataset.py \
    --max_num_examples $max_num_examples \
    --success_threshold $success_threshold \
    --judgments_name $judgments_name

done
done

# Create SFT data based on gpt-4.1-nano judge

judgments_name="judgments"

for max_num_examples in 2000 5000 10000 20000; do
for success_threshold in 0.5; do

python ~/insta-dev/sft/create_sft_dataset.py \
    --max_num_examples $max_num_examples \
    --success_threshold $success_threshold \
    --judgments_name $judgments_name

done
done
