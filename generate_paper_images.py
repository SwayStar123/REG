import argparse
import math
import os

import numpy as np
import torch
from PIL import Image

from models.sit import SiT_models
from preprocessing.encoders import load_invae
from samplers import euler_maruyama_sampler, euler_maruyama_sampler_path_drop
from utils import load_legacy_checkpoints, download_model


def build_experiment_folder(args):
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-invae-"
        f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-{args.guidance_high}-"
        f"{args.cls_cfg_scale}-pathdrop-{args.path_drop}"
    )
    if args.balanced_sampling:
        folder_name += "-balanced"
    paper_dir = os.path.join(args.sample_dir, "paper_images")
    os.makedirs(paper_dir, exist_ok=True)
    return paper_dir


def load_models(args, device):
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 16
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=32,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(",")],
        **block_kwargs,
    ).to(device)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = "SiT-XL-2-256x256.pt"
        assert args.model == "SiT-XL/2"
        assert len(args.projector_embed_dims.split(",")) == 1
        assert int(args.projector_embed_dims.split(",")[0]) == 768
        state_dict = download_model("last.pt")
    else:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)["ema"]

    if args.legacy:
        state_dict = load_legacy_checkpoints(state_dict=state_dict, encoder_depth=args.encoder_depth)
    model.load_state_dict(state_dict)
    model.eval()

    vae = load_invae("REPA-E/e2e-invae").to(device)
    vae.eval().requires_grad_(False)

    return model, vae, latent_size


def sample_images_for_label(args, model, vae, latent_size, device, label_id):
    # 2 full-res, 4 half-res row, 8 quarter-res row
    n_top = 2
    n_mid = 4
    n_bot = 8
    n_total = n_top + n_mid + n_bot

    z = torch.randn(n_total, model.in_channels, latent_size, latent_size, device=device)
    y = torch.full((n_total,), int(label_id), device=device, dtype=torch.long)
    cls_z = torch.randn(n_total, args.cls, device=device)

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
        args=args,
    )

    with torch.no_grad():
        if args.mode == "sde":
            if args.path_drop:
                samples = euler_maruyama_sampler_path_drop(**sampling_kwargs).to(torch.float32)
            else:
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
        elif args.mode == "ode":
            raise NotImplementedError("ODE mode is not implemented in this helper script.")
        else:
            raise NotImplementedError()

        scaling_factor = 0.3099
        samples = vae.decode(samples / scaling_factor).sample
        samples = (samples + 1) / 2.0
        samples = torch.clamp(255.0 * samples, 0, 255).permute(0, 2, 3, 1).to(
            "cpu", dtype=torch.uint8
        ).numpy()

    pil_images = [Image.fromarray(s) for s in samples]
    return pil_images


def make_paper_style_grid(images, resolution):
    # images: list of 14 PIL images
    assert len(images) == 14, "Expected 14 images (2 + 4 + 8)."
    res = resolution
    top_h = res
    mid_h = res // 2
    bot_h = res // 4

    canvas_w = res * 2
    canvas_h = top_h + mid_h + bot_h
    canvas = Image.new("RGB", (canvas_w, canvas_h))

    # Top row: 2 images at full resolution
    for i in range(2):
        img = images[i].resize((res, res), Image.BICUBIC)
        x = i * res
        y = 0
        canvas.paste(img, (x, y))

    # Middle row: 4 images at half resolution
    for i in range(4):
        img = images[2 + i].resize((res // 2, res // 2), Image.BICUBIC)
        x = i * (res // 2)
        y = top_h
        canvas.paste(img, (x, y))

    # Bottom row: 8 images at quarter resolution
    for i in range(8):
        img = images[6 + i].resize((res // 4, res // 4), Image.BICUBIC)
        x = i * (res // 4)
        y = top_h + mid_h
        canvas.paste(img, (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser()

    # seed / precision
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TF32 matmuls (Ampere GPUs).",
    )

    # logging/saving
    parser.add_argument("--ckpt", type=str, default="exps/b1-reg-invae-sprint-rms-rope-qknorm-valres-cfm-timeshifting/checkpoints/0400000.pt", help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="exps/b1-reg-invae-sprint-rms-rope-qknorm-valres-cfm-timeshifting")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/1")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)

    # sampling hyperparameters (single-GPU)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--cls-cfg-scale", type=float, default=2.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.2)
    parser.add_argument("--guidance-high", type=float, default=0.8)
    parser.add_argument("--cls", default=768, type=int)
    parser.add_argument("--path-drop", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-shifting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shift-base", type=int, default=4096)

    parser.add_argument(
        "--num-random-classes",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    assert torch.cuda.is_available(), "This script requires at least one GPU."
    device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device)

    paper_dir = build_experiment_folder(args)
    print(f"Saving paper-style images to: {paper_dir}")

    model, vae, latent_size = load_models(args, device)

    num_to_sample = min(args.num_random_classes, args.num_classes)
    label_ids = torch.randperm(args.num_classes)[:num_to_sample].tolist()

    for label_id in label_ids:
        print(f"Generating paper-style grid for label {label_id}...")
        images = sample_images_for_label(args, model, vae, latent_size, device, label_id)
        grid = make_paper_style_grid(images, args.resolution)
        out_path = os.path.join(paper_dir, f"{int(label_id)}.png")
        grid.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
