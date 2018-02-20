#/bin/bash

trap 'pkill -P $$' EXIT

while getopts ":c" opt;do
	case $opt in
		c)
			echo "Clearing previous data"
			rm logs/*
			rm -r data/*
			rm -r saved_model
			;;
		\?)
			echo "INVALID OPTION: -$OPTARG"	 
			;;
	esac
done

n_worker=4

ngames=1

n_sims=250

DATA_PATHS=""
for ((i=1; i<=${n_worker}; i++)){
	mkdir -p data/self${i}
	DATA_PATHS+=" data/self${i}" 
}


for ((x=1; x<200; x++)){
	echo Cycle $x 
	python3 train.py --new --save_loss --batch_size 32 --max_iters 200000 --epochs 25 --data_dir $DATA_PATHS --shuffle --val_split 0.01 >> logs/log_train

	for ((i=1; i<=$n_worker; i++)){
		python3 play.py --cycle $x --selfplay --ngames $ngames --mcts_sims $n_sims --save --save_dir data/self$i/ >> logs/log_$i &
	}
	
	wait
}

