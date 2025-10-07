NUM_GPUS=8
random_number=$((RANDOM % 100 + 1200))


accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="bf16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/1" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --output-dir="exps" \
    --exp-name="xl1-reg-invae-tcfm" \
    --batch-size=256 \
    --data-dir="dataset"


    #Dataset Path
    #For example: your_path/imagenet-vae
    #This folder contains two folders
    #(1) The imagenet's RGB image: your_path/imagenet-vae/imagenet_256-vae/
    #(2) The imagenet's VAE latent: your_path/imagenet-vae/vae-sd/