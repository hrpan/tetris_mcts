#/bin/bash

trap 'pkill -P $$' EXIT

curr_cycle=1

#TRAIN ARGS
last_nfiles=1
n_worker=1
val_mode=0
val_eps=10
batch_size=128
epochs=10

#PLAY ARGS
ngames=100
agent_type=ValueSim
n_sims=500
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
    #DATA_PATHS+=" data/self${i}/*" 
    DATA_PATHS+=" data/self${i}/tree*" 
}
mkdir -p data/benchmark

for ((x=$curr_cycle; x<200; x++)){
    echo Cycle $x 
    python train.py --td \
        --weighted_mse \
        --weighted_mse_mode 1 \
        --save_loss \
        --batch_size $batch_size \
        --epochs $epochs \
        --data_paths $DATA_PATHS \
        --validation \
        --val_episodes $val_eps \
        --val_mode $val_mode \
        --val_total 100 \
        --last_nfiles $last_nfiles \
        --save_interval 250 >> logs/log_train 2>> logs/log_err

    for ((i=1; i<=$n_worker; i++)){
        python play.py --agent_type $agent_type --cycle $x --selfplay --ngames $ngames --mcts_sims $n_sims --save --save_dir data/self$i/ >> logs/log_$i 2>> logs/log_err &
    }
    python play.py --agent_type $agent_type --cycle $x --selfplay --ngames 1 --mcts_sims $n_sims_bench --save --save_dir data/benchmark/ >> logs/log_benchmark 2>> logs/log_err
    wait
}

