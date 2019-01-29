#/bin/bash

trap 'pkill -P $$' EXIT

curr_cycle=1

#TRAIN ARGS
last_nfiles=1
n_worker=1
val_eps=5

#PLAY ARGS
ngames=250
agent_type=ValueSimBayes
n_sims=300
n_sims_bench=1000

while getopts ":cr" opt;do
    case $opt in
        c)
            echo "Clearing previous data"
            rm logs/*
            rm -r data/*
            rm -r saved_model
            rm -r pytorch_model
            ;;
        r)
            echo "Returning to previous cycles"
            last_cycle="$(ls data/self1/ | tr -d 'data' | sort -n | tail -n 1)" 
            curr_cycle=$((last_cycle+1))
            ;;
        \?)
            echo "INVALID OPTION: -$OPTARG"  
            ;;
    esac
done

DATA_PATHS=""
for ((i=1; i<=${n_worker}; i++)){
    mkdir -p data/self${i}
    DATA_PATHS+=" data/self${i}/*" 
}
mkdir -p data/benchmark

for ((x=$curr_cycle; x<200; x++)){
    echo Cycle $x 
    python train.py --td \
        --target_normalization \
        --save_loss \
        --batch_size 32 \
        --epochs 10 \
        --data_paths $DATA_PATHS \
        --val_episodes $val_eps \
        --last_nfiles $last_nfiles \
        --val_total 100 \
        --save_interval 250 >> logs/log_train 2>> logs/log_err

    for ((i=1; i<=$n_worker; i++)){
        python play.py --agent_type $agent_type --cycle $x --selfplay --ngames $ngames --mcts_sims $n_sims --save --save_dir data/self$i/ >> logs/log_$i 2>> logs/log_err &
    }
    python play.py --agent_type $agent_type --cycle $x --selfplay --ngames 1 --mcts_sims $n_sims_bench --save --save_dir data/benchmark/ >> logs/log_benchmark 2>> logs/log_err
    wait
}

