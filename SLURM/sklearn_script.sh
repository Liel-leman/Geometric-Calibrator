#! /bin/bash
metric="none"
calibrated=0

if [ $calibrated -eq 1 ]
then
  isCali="c"
else
  isCali=""
fi


declare -a datasets=("CIFAR_RGB" "Fashion")
for dataset in "${datasets[@]}"
	do
			declare -a models=("GB" "RF")
			for model in "${models[@]}"
				do
			for i in {0..9}
				do
					python run_shuffle_on_data_model.py ${dataset} ${model} ${i} ${metric} ${calibrated}
                    
				done
			done
	done
