#! /bin/bash


norm="L2"
START=0
END=99
#myArray=("78")
model="cnn"


#declare -a arr_data=("SignLanguage" "MNIST" "Fashion" "CIFAR_RGB" "GTSRB_RGB")
declare -a arr_data=("SignLanguage")
	for dataset in "${arr_data[@]}"
		do
		
		if [ $dataset == "SignLanguage" ]
		then
			RAM="10"
		elif [ $dataset == "MNIST" ] || [ $dataset == "Fashion" ]
		then
			RAM="10"
		elif [ $dataset == "CIFAR_RGB" ]
		then
			RAM="20"
		elif [ $dataset == "GTSRB_RGB" ]
		then
			RAM="30"
		fi
		
	#for i in ${myArray[@]}
	for i in $(eval echo "{$START..$END}")
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
			#SBATCH --job-name ${i}CNN_${dataset}		### name of the job
			#SBATCH --output=/home/leman/Sep/cluster/outputFiles/$CNN_${dataset}_${i}_%J.out		### output log for running job - %J for job number
			#SBATCH --ntasks=4
			#SBATCH --cpus-per-task=8

			# Note: the following 4 lines are commented out
			#SBATCH --mail-user=leman@post.bgu.ac.il	### user's email for sending job status messages
			#SBATCH --mail-type=ALL				### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
			#SBATCH --mem=${RAM}G				### ammount of RAM memory

			### Print some data to output file ###
			echo `date`
			echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
			echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

			### Start your code below ####
			module load anaconda				### load anaconda module (must be present when working with conda environments)
			source activate ML				### activate a conda environment, replace my_env with your conda environment
			python pytorchShuffle.py ${dataset} ${i} ${norm}	### execute python script – replace with your own command 
			EOF
			
			sbatch $File 
		done
	done
