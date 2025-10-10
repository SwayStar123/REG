# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Licensed under The MIT License [see LICENSE for details]
# Modified: add rope_ids support for routed / unsorted token subsets
# --------------------------------------------------------

from math import pi
from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), \
        'invalid dimensions for broadcastable concatenation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    # last dim must be even
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    """
    Original EVA-02 2D RoPE (non-fast) with a start_index window.
    Kept unchanged for compatibility with other code paths.
    """
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], \
            f'feature dimension {t.shape[-1]} is not sufficient to rotate {rot_dim} positions'
        t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t_mid = (t_mid * self.freqs_cos.to(dtype=t.dtype)) + (rotate_half(t_mid) * self.freqs_sin.to(dtype=t.dtype))
        return torch.cat((t_left, t_mid, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    """
    Fast EVA-02 2D RoPE with broadcasting, extended to support per-token rope_ids.
    Shapes:
      - Precomputed buffers:
          freqs_cos, freqs_sin: (HW, D_rot), where H=W=pt_seq_len, D_rot=2*dim
      - Typical attention tensors:
          q, k: (B, Hh, N, D_rot) where Hh = num_heads, N = HW or a routed N_keep
      - rope_ids (optional):
          None           -> assume N == HW and use default [0..HW-1] order
          LongTensor(N,) -> shared positions for all batches (rare)
          LongTensor(B,N)-> per-batch positions (used for routing with unsorted ids_keep)
    """
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        # Build 2D grid freqs of shape (H, W, 2*dim), then flatten to (HW, 2*dim)
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        base = torch.einsum('..., f -> ... f', t, freqs)           # (S, dim//2)
        base = repeat(base, '... n -> ... (n r)', r=2)             # (S, dim)
        freqs_2d = broadcat((base[:, None, :], base[None, :, :]), dim=-1)  # (S, S, 2*dim)

        freqs_cos = freqs_2d.cos().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)
        freqs_sin = freqs_2d.sin().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # keep for sanity checks or downstream use
        self.grid_size = ft_seq_len  # H == W == grid_size
        self.rot_dim = freqs_2d.shape[-1]  # 2*dim

    def _gather_cos_sin(
        self,
        rope_ids: Optional[torch.Tensor],
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Returns cos, sin shaped for broadcasting over (B, Hh, N, D).
        - If rope_ids is None: shapes (1,1,N,D), assuming N==HW (standard full-grid order).
        - If rope_ids is (N,): shapes (1,1,N,D) with custom order shared by all batches.
        - If rope_ids is (B,N): shapes (B,1,N,D) with per-batch custom order.
        """
        cos_table = self.freqs_cos.to(dtype=dtype, device=device)
        sin_table = self.freqs_sin.to(dtype=dtype, device=device)

        if rope_ids is None:
            # Default sequential positions [0..HW-1], requires N == HW
            assert N == cos_table.shape[0], \
                f"When rope_ids is None, expected N == HW ({cos_table.shape[0]}), got N={N}"
            cos = cos_table.view(1, 1, N, -1)
            sin = sin_table.view(1, 1, N, -1)
            return cos, sin

        # Ensure long dtype for indexing
        rope_ids = rope_ids.to(device=device, dtype=torch.long)

        if rope_ids.dim() == 1:
            # (N,)
            cos = cos_table.index_select(0, rope_ids).view(1, 1, N, -1)
            sin = sin_table.index_select(0, rope_ids).view(1, 1, N, -1)
            return cos, sin

        if rope_ids.dim() == 2:
            # (B, N) — per-batch indexing; gather rows per batch
            # Advanced indexing will produce (B, N, D)
            cos = cos_table[rope_ids].unsqueeze(1)  # (B, 1, N, D)
            sin = sin_table[rope_ids].unsqueeze(1)  # (B, 1, N, D)
            return cos, sin

        raise ValueError(f"rope_ids must be None, (N,), or (B,N); got shape {tuple(rope_ids.shape)}")

    def forward(self, t: torch.Tensor, rope_ids: Optional[torch.Tensor] = None):
        """
        t: (B, Hh, N, D_rot) where D_rot == self.rot_dim
        rope_ids: None | (N,) | (B, N), indexing flattened HW positions
        """
        B, Hh, N, D = t.shape
        assert D == self.rot_dim, \
            f"Head dim {D} must equal RoPE rotation dim {self.rot_dim} (got {D} != {self.rot_dim})"

        cos, sin = self._gather_cos_sin(rope_ids, N, t.device, t.dtype)  # shapes (1 or B, 1, N, D)
        # Broadcast over heads; cos/sin already have (1 or B, 1, N, D), so broadcasting to (B, Hh, N, D) is natural
        return t * cos + rotate_half(t) * sin