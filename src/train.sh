export GPU_ID="4,5,6,7"
export STORE_PATH="./output/Desc2ProgramGeo/"
export BATCH_SIZE=16
export SEED=123
export N_WORKERS=4
export MAX_WORKERS=35
export GRAPH_HIDDEN=64
export EMBEDDING_DIM=300
export SKETCH_HIDDEN=256
export PROGRAM_HIDEEN=256
export TRAIN_ITERS=25000
export MODEL_LR_RATE=0.0003

python ./sketc2prog_llm_train.py --gpu_id=$GPU_ID --store_path=$STORE_PATH --batch_size=$BATCH_SIZE --seed=$SEED --n_workers=$N_WORKERS --max_words=$MAX_WORKERS --graph_hidden=$GRAPH_HIDDEN --embedding_dim=$EMBEDDING_DIM --sketch_hidden=$SKETCH_HIDDEN --program_hidden=$PROGRAM_HIDEEN --train_iters=$TRAIN_ITERS --model_lr_rate=$MODEL_LR_RATE