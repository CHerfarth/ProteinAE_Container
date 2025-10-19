#!/bin/bash
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

module load singularity

mkdir output

singularity exec --nv \
	-B output:/proteinae/output \
	-B configs:/proteinae/configs \
	ProteinAE.sif \
	python3 /proteinae/proteinfoundation/autoencode.py \
	--input_pdb /proteinae/examples/7v11.pdb \
	--output_dir /proteinae/output \
	--config_path /proteinae/configs \
	--mode autoencode
