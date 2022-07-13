#! /bin/bash


declare -a datasets=( "MNIST" "Fashion" "SignLanguage")
for dataset in "${datasets[@]}"
	do
			declare -a models=("GB" "RF")
			for model in "${models[@]}"
				do
			for i in {0..9}
				do
					File="sbatch_${dataset}_${model}_${i}.example"
					if [ ! -e "$File" ]; then              #Check if file exists
						echo "Creating file $File"
						touch $File                          #Create file if it doesn't exist
					fi 
					
					cat <<- EOF > $File
					#!/bin/bash
					################################################################################################
					### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
					### To ignore, just add another # - like so: ##SBATCH
					################################################################################################
					#SBATCH --partition main			### specify partition name where to run a job. short: 7 days limit; gtx1080: 7 days; debug: 2 hours limit and 1 job at a time
					#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
					#SBATCH --job-name ${model}${i}_${dataset}	### name of the job
					#SBATCH --output=/home/chouraga/Sep/cluster/outputFiles/${model}_${dataset}_${i}_%J.out		### output log for running job - %J for job number
					#SBATCH --ntasks=1
					#SBATCH --cpus-per-task=1
					
					#SBATCH --mem=10G				### ammount of RAM memory
					#SBATCH --nodes=1				###
					
					### Start your code below ####
					module load anaconda				### load anaconda module (must be present when working with conda environments)
					source activate ML				### activate a conda environment, replace my_env with your conda environment
					python all_robs.py ${dataset} ${model} ${i} 	### execute python script – replace with your own command 
					EOF
					
					
					sbatch $File 
				done
			done
        done