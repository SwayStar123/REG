NUM_GPUS=8
random_number=$((RANDOM % 100 + 1200))


accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --output-dir="exp" \
    --exp-name="xl2_reg_sra_dispersive" \
    --batch-size=256 \
    --data-dir="dataset/" \
    --cls=0.03 \
    --no-use-redi \
    --use-sra \
    --use-dispersive-loss
