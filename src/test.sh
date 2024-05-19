export GPU_ID="4,"
export CHECKPOINT="./output/Desc2ProgramGeo/desc2program-best.ckpt"
export BATCH_SIZE=16

python ./sketc2prog_llm_test.py --gpu_id=$GPU_ID --checkpoint=$CHECKPOINT --batch_size=$BATCH_SIZE