# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

from collections.abc import Iterator
from dataclasses import dataclass
import io
import json
import multiprocessing as mp
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from encoders import StabilityVAEEncoder

#----------------------------------------------------------------------------

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

#----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    input_images = []
    def _recurse_dirs(root: str): # workaround Path().rglob() slowness
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname).convert('RGB'))
            yield ImageEntry(img=img, label=labels.get(arch_fnames[fname]))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------



#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

def scale_image(width, height, img):
    """Scale image to specified dimensions."""
    w = img.shape[1]
    h = img.shape[0]
    if width == w and height == h:
        return img
    img = PIL.Image.fromarray(img, 'RGB')
    ww = width if width is not None else w
    hh = height if height is not None else h
    img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
    return np.array(img)

def center_crop_image(width, height, img):
    """Center crop and resize image."""
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    return np.array(img)

def center_crop_wide_image(width, height, img):
    """Center crop wide image."""
    ch = int(np.round(width * img.shape[0] / img.shape[1]))
    if img.shape[1] < width or ch < height:
        return None

    img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    img = np.array(img)

    canvas = np.zeros([width, width, 3], dtype=np.uint8)
    canvas[(width - height) // 2 : (width + height) // 2, :] = img
    return canvas

def center_crop_imagenet_image(image_size, arr):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = PIL.Image.fromarray(arr)
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    assert len(new_size) == 2
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

def transform_image_worker(args):
    """Worker function for parallel image transformation."""
    idx, image_data, transform_type, output_width, output_height = args
    try:
        img = image_data.img
        
        if transform_type is None:
            img = scale_image(output_width, output_height, img)
        elif transform_type == 'center-crop':
            img = center_crop_image(output_width, output_height, img)
        elif transform_type == 'center-crop-wide':
            img = center_crop_wide_image(output_width, output_height, img)
        elif transform_type == 'center-crop-dhariwal':
            img = center_crop_imagenet_image(output_width, img)
        else:
            raise ValueError(f'Unknown transform type: {transform_type}')
            
        if img is None:
            return None
        return idx, img, image_data.label
    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        return None

def encode_image_worker(args):
    """Worker function for parallel VAE encoding."""
    gpu_id, batch_data, model_url = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=1)
    results = []
    
    for idx, image_data in batch_data:
        try:
            img_tensor = torch.tensor(image_data.img).to('cuda').permute(2, 0, 1).unsqueeze(0)
            mean_std = vae.encode_pixels(img_tensor)[0].cpu().numpy()
            results.append((idx, mean_std, image_data.label))
        except Exception as e:
            print(f"Error encoding image {idx} on GPU {gpu_id}: {e}")
            results.append(None)
    
    return results

#----------------------------------------------------------------------------

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)
@click.option('--workers',    help='Number of parallel workers for image processing', metavar='INT', type=int, default=32, show_default=True)
@click.option('--batch-size', help='Number of images to process in each batch', metavar='INT', type=int, default=1000, show_default=True)

def convert(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    workers: int,
    batch_size: int
):
    """Convert an image dataset into archive format for training.

    Specifying the input images:

    \b
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Parallelization and Memory Management:

    Use --workers to control the number of CPU cores used for image processing
    (default: 32). Use --batch-size to control how many images are processed
    in each batch (default: 1000). Larger batch sizes use more memory but may
    be more efficient. For very large datasets (like ImageNet), consider using
    smaller batch sizes to avoid memory issues.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    The --transform=center-crop-dhariwal selects a crop/rescale mode that is intended
    to exactly match with results obtained for ImageNet in common diffusion model literature:

    \b
    python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \\
        --dest=datasets/img64.zip --resolution=64x64 --transform=center-crop-dhariwal \\
        --workers=32
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    
    # Validate transform parameters
    if transform in ['center-crop', 'center-crop-wide', 'center-crop-dhariwal']:
        if resolution is None:
            raise click.ClickException(f'must specify --resolution=WxH when using {transform} transform')
        if transform == 'center-crop-dhariwal' and resolution[0] != resolution[1]:
            raise click.ClickException('width and height must match in --resolution=WxH when using center-crop-dhariwal transform')
    
    dataset_attrs = None

    # Process images in batches to avoid loading everything into memory
    output_width, output_height = resolution if resolution is not None else (None, None)
    labels = []
    
    print(f"Processing {num_files} images in batches of {batch_size} with {workers} workers...")
    
    with mp.Pool(workers) as pool:
        batch = []
        batch_start_idx = 0
        
        for idx, image in tqdm(enumerate(input_iter), total=num_files, desc="Processing images"):
            batch.append((idx, image, transform, output_width, output_height))
            
            # Process batch when it's full or we've reached the end
            if len(batch) == batch_size or idx == num_files - 1:
                # Process current batch in parallel
                batch_results = pool.map(transform_image_worker, batch)
                
                # Filter valid results and process them
                valid_batch_results = [result for result in batch_results if result is not None]
                valid_batch_results.sort(key=lambda x: x[0])  # Sort by index
                
                # Save results from this batch
                for result_idx, img, label in valid_batch_results:
                    idx_str = f'{result_idx:08d}'
                    archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

                    # Error check to require uniform image attributes across
                    # the whole dataset.
                    assert img.ndim == 3
                    cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0]}
                    if dataset_attrs is None:
                        dataset_attrs = cur_image_attrs
                        width = dataset_attrs['width']
                        height = dataset_attrs['height']
                        if width != height:
                            raise click.ClickException(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
                        if width != 2 ** int(np.floor(np.log2(width))):
                            raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
                    elif dataset_attrs != cur_image_attrs:
                        err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
                        raise click.ClickException(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

                    # Save the image as an uncompressed PNG.
                    img = PIL.Image.fromarray(img)
                    image_bits = io.BytesIO()
                    img.save(image_bits, format='png', compress_level=0, optimize=False)
                    save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
                    labels.append([archive_fname, label] if label is not None else None)
                
                # Clear batch for next iteration
                batch = []

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--gpus',       help='Number of GPUs to use for parallel encoding', metavar='INT', type=int, default=8, show_default=True)
@click.option('--batch-size', help='Number of images per GPU in each batch', metavar='INT', type=int, default=100, show_default=True)

def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    gpus: int,
    batch_size: int,
):
    """Encode pixel data to VAE latents.
    
    Parallelization and Memory Management:
    
    Use --gpus to control the number of GPUs used for parallel encoding
    (default: 8). Use --batch-size to control how many images each GPU
    processes in each batch (default: 100). Images are distributed across
    GPUs in round-robin fashion and processed in batches to avoid loading
    all images into memory at once.
    
    Example:
    \b
    python dataset_tool.py encode --source=datasets/img64.zip \\
        --dest=datasets/img64_encoded.zip --gpus=8 --batch-size=50
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    
    # Process images in batches across GPUs to avoid loading everything into memory
    labels = []
    
    print(f"Processing {num_files} images in batches of {batch_size * gpus} across {gpus} GPUs...")
    
    with mp.Pool(gpus) as pool:
        batch = []
        gpu_batches = [[] for _ in range(gpus)]
        
        for idx, image in tqdm(enumerate(input_iter), total=num_files, desc="Encoding images"):
            # Distribute images across GPUs in round-robin fashion
            gpu_id = idx % gpus
            gpu_batches[gpu_id].append((idx, image))
            
            # Process when any GPU batch is full or we've reached the end
            max_batch_size = max(len(gpu_batch) for gpu_batch in gpu_batches)
            if max_batch_size >= batch_size or idx == num_files - 1:
                # Prepare arguments for each GPU
                gpu_args = []
                for gpu_id, gpu_batch in enumerate(gpu_batches):
                    if gpu_batch:  # Only process non-empty batches
                        gpu_args.append((gpu_id, gpu_batch, model_url))
                
                # Process current batches in parallel across GPUs
                if gpu_args:
                    batch_results = pool.map(encode_image_worker, gpu_args)
                    
                    # Flatten results and sort by index
                    current_results = []
                    for gpu_result in batch_results:
                        current_results.extend([r for r in gpu_result if r is not None])
                    current_results.sort(key=lambda x: x[0])
                    
                    # Save results from this batch
                    for result_idx, mean_std, label in current_results:
                        idx_str = f'{result_idx:08d}'
                        archive_fname = f'{idx_str[:5]}/img-mean-std-{idx_str}.npy'

                        f = io.BytesIO()
                        np.save(f, mean_std)
                        save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
                        labels.append([archive_fname, label] if label is not None else None)
                
                # Clear batches for next iteration
                gpu_batches = [[] for _ in range(gpus)]

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    cmdline()

#----------------------------------------------------------------------------