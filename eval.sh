
random_number=$((RANDOM % 100 + 1200))
NUM_GPUS=8
STEP="0800000"
SAVE_PATH="exps/l16-pixelnerd_type_thing"
NUM_STEP=250
MODEL_SIZE='L'
CFG_SCALE=3.5
CLS_CFG_SCALE=1.0
GH=1.0


export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch --master_port=$random_number --nproc_per_node=$NUM_GPUS generate.py \
  --model SiT-${MODEL_SIZE}/16 \
  --num-fid-samples 50000 \
  --ckpt ${SAVE_PATH}/checkpoints/${STEP}.pt \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=${NUM_STEP} \
  --cfg-scale=${CFG_SCALE} \
  --cls-cfg-scale=${CLS_CFG_SCALE} \
  --guidance-high=${GH} \
  --sample-dir ${SAVE_PATH}/checkpoints \
  --cls=768 


python ./evaluations/evaluator.py \
    --ref_batch evaluations/VIRTUAL_imagenet256_labeled.npz \
    --sample_batch ${SAVE_PATH}/checkpoints/SiT-${MODEL_SIZE}-16-${STEP}-size-256-cfg-${CFG_SCALE}-seed-0-sde-${GH}-${CLS_CFG_SCALE}.npz \
    --save_path ${SAVE_PATH}/checkpoints \
    --cfg_cond 1 \
    --step ${STEP} \
    --num_steps ${NUM_STEP} \
    --cfg ${CFG_SCALE} \
    --cls_cfg ${CLS_CFG_SCALE} \
    --gh ${GH}











