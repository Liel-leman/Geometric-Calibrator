#! /bin/bash


declare -a datasets=("MNIST" )
for dataset in "${datasets[@]}"
	do
        python_file="pytorchShuffle${dataset}.py"
		for i in {0..9}
			do
				python ${python_file} ${dataset} ${i}
                    
			done
		done
done
