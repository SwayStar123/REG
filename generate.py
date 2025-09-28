# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def count_existing_samples(sample_folder_dir):
    """
    Count existing .png files in the sample folder.
    """
    if not os.path.exists(sample_folder_dir):
        return 0
    
    png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
    return len(png_files)


def find_missing_indices(sample_folder_dir, num_fid_samples):
    """
    Find which image indices are missing from the sample folder.
    Returns a set of missing indices.
    """
    if not os.path.exists(sample_folder_dir):
        return set(range(num_fid_samples))
    
    existing_indices = set()
    png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
    
    for filename in png_files:
        try:
            # Extract index from filename like "046598.png"
            index = int(filename.replace('.png', ''))
            if 0 <= index < num_fid_samples:
                existing_indices.add(index)
        except ValueError:
            # Skip files that don't follow the expected naming pattern
            continue
    
    # Return missing indices
    all_indices = set(range(num_fid_samples))
    missing_indices = all_indices - existing_indices
    return missing_indices


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Create folder name first to check for existing samples
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-{args.guidance_high}-{args.cls_cfg_scale}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    # Check if we already have enough samples
    existing_samples = count_existing_samples(sample_folder_dir)
    missing_indices = find_missing_indices(sample_folder_dir, args.num_fid_samples)
    
    if rank == 0:
        print(f"Found {existing_samples} existing samples in {sample_folder_dir}")
        print(f"Missing {len(missing_indices)} samples")
    
    if len(missing_indices) == 0:
        if rank == 0:
            print(f"All {args.num_fid_samples} samples already exist. Skipping generation, creating npz...")
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()
        return
    
    # Convert missing indices to a sorted list for distributed generation
    missing_indices_list = sorted(list(missing_indices))
    samples_needed = len(missing_indices_list)
    if rank == 0:
        print(f"Need to generate {samples_needed} missing samples")

    if args.space == "latent":
        latent_size = args.resolution // 8
        channels = 4
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
        vae.eval()

        latents_scale = torch.tensor(
            [0.18215, 0.18215, 0.18215, 0.18215, ]
            ).view(1, 4, 1, 1).to(device)
        latents_bias = -torch.tensor(
            [0., 0., 0., 0.,]
            ).view(1, 4, 1, 1).to(device)
    else:
        latent_size = args.resolution
        channels = 3
        vae = None
    

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        in_channels=channels,
        subpatch_size=args.subpatch_size,
        uvit_skips=args.uvit_skips,
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['ema']

    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    model.eval()  # important!

    # Create folder to save samples:
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(samples_needed / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    
    # Create a queue of missing indices for each GPU to process
    indices_per_gpu = [[] for _ in range(dist.get_world_size())]
    for i, idx in enumerate(missing_indices_list):
        gpu_rank = i % dist.get_world_size()
        indices_per_gpu[gpu_rank].append(idx)
    
    # Get the indices this GPU should process
    my_indices = indices_per_gpu[rank]
    
    # Pad with dummy indices if needed to make batches even
    while len(my_indices) % n != 0:
        my_indices.append(-1)  # Use -1 as dummy index
    
    my_batches = [my_indices[i:i+n] for i in range(0, len(my_indices), n)]
    total_batch_idx = 0

    
    for batch_indices in my_batches:
        # Get the current batch of indices to generate
        current_batch = [idx for idx in batch_indices if idx != -1]  # Filter out dummy indices
        if not current_batch:  # Skip if all indices are dummy
            continue
            
        # Set deterministic seed for each specific image index to ensure reproducibility
        batch_size = len(current_batch)
        z = torch.zeros(batch_size, channels, latent_size, latent_size, device=device)
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        cls_z = torch.zeros(batch_size, args.cls, device=device)
        
        # Generate deterministic samples for each index
        for i, idx in enumerate(current_batch):
            # Use the image index as part of the seed for deterministic generation
            torch.manual_seed(args.global_seed + idx)
            z[i] = torch.randn(channels, latent_size, latent_size, device=device)
            y[i] = torch.randint(0, args.num_classes, (1,), device=device)
            cls_z[i] = torch.randn(args.cls, device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            cls_latents=cls_z,
            args=args
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":# will support
                exit()
                #samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            if args.space == "latent":
                samples = vae.decode((samples -  latents_bias) / latents_scale).sample

            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk using the specific indices
            for i, sample in enumerate(samples):
                if i < len(current_batch):  # Make sure we don't exceed the current batch
                    index = current_batch[i]
                    Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--uvit-skips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subpatch-size", type=int, default=4)
    parser.add_argument("--space", type=str, default="latent", choices=["pixel", "latent"])

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cls-cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--cls', default=768, type=int)
    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode


    args = parser.parse_args()
    main(args)
