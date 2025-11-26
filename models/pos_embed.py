# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

from math import pi
from typing import Optional

import torch
from torch import nn

from einops import rearrange, repeat

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    """
    Fast EVA-02 2D RoPE with broadcasting, extended to support:
      • per-token rope_ids (routing / unsorted subsets)
      • leading CLS tokens (unrotated)
    Accepts q/k shaped (B, Hh, N, D_rot) where N may be HW or HW+extra.
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

        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        base = torch.einsum('..., f -> ... f', t, freqs)           # (S, dim//2)
        base = repeat(base, '... n -> ... (n r)', r=2)             # (S, dim)
        freqs_2d = broadcat((base[:, None, :], base[None, :, :]), dim=-1)  # (S,S,2*dim)

        freqs_cos = freqs_2d.cos().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)
        freqs_sin = freqs_2d.sin().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        self.grid_size = ft_seq_len                   # H == W
        self.rot_dim = freqs_2d.shape[-1]            # 2*dim

    def _gather_cos_sin(self, rope_ids, N, device, dtype):
        cos_table = self.freqs_cos.to(dtype=dtype, device=device)
        sin_table = self.freqs_sin.to(dtype=dtype, device=device)

        if rope_ids is None:
            assert N == cos_table.shape[0], \
                f"When rope_ids is None, expected N == HW ({cos_table.shape[0]}), got N={N}"
            return cos_table.view(1, 1, N, -1), sin_table.view(1, 1, N, -1)

        rope_ids = rope_ids.to(device=device, dtype=torch.long)

        if rope_ids.dim() == 1:
            cos = cos_table.index_select(0, rope_ids).view(1, 1, N, -1)
            sin = sin_table.index_select(0, rope_ids).view(1, 1, N, -1)
            return cos, sin

        if rope_ids.dim() == 2:
            cos = cos_table[rope_ids].unsqueeze(1)  # (B,1,N,D)
            sin = sin_table[rope_ids].unsqueeze(1)  # (B,1,N,D)
            return cos, sin

        raise ValueError(f"rope_ids must be None, (N,), or (B,N); got {tuple(rope_ids.shape)}")

    def forward(self, t: torch.Tensor, rope_ids: Optional[torch.Tensor] = None):
        """
        t: (B, Hh, N, D_rot), D_rot == self.rot_dim
        rope_ids: None | (N,) | (B,N), indexing flattened HW positions for the
                  rotated portion. If t includes CLS, supply rope_ids for the
                  spatial tail only or for the full N; both are accepted.
        """
        B, Hh, N, D = t.shape
        assert D == self.rot_dim, f"Head dim {D} must equal RoPE dim {self.rot_dim}"

        HW = self.freqs_cos.shape[0]

        # Determine how many leading tokens to leave unrotated (CLS or others).
        if rope_ids is None:
            # No ids -> sequence must be either [HW] or [extra + HW] in default grid order.
            if N == HW:
                extra = 0
            else:
                assert N >= HW, f"N={N} shorter than HW={HW}"
                extra = N - HW
        else:
            # ids given for either full N or just the spatial tail.
            ids_len = rope_ids.shape[-1]
            if ids_len == N:
                extra = max(0, N - HW)    # assume any surplus is leading CLS tokens
            else:
                # ids describe only the spatial tail
                extra = N - ids_len
                assert extra >= 0, "rope_ids longer than sequence length"
                # quick sanity: spatial tail size should match HW when not routing
                # but allow routing to keep arbitrary N_tail
            # Ensure the tail length we will rotate is positive
        assert extra <= N, "extra leading tokens exceeds sequence length"

        if extra > 0:
            t_lead = t[:, :, :extra, :]            # unrotated CLS or similar
            t_tail = t[:, :, extra:, :]            # rotate these
            ids_tail = None if rope_ids is None else \
                       (rope_ids if rope_ids.shape[-1] == t_tail.shape[-2] else rope_ids[..., extra:])
        else:
            t_lead = None
            t_tail = t
            ids_tail = rope_ids

        if t_tail.shape[-2] == 0:
            # Only CLS present
            return t

        cos, sin = self._gather_cos_sin(ids_tail, t_tail.shape[-2], t.device, t.dtype)
        rot = t_tail * cos + rotate_half(t_tail) * sin

        return rot if t_lead is None else torch.cat([t_lead, rot], dim=-2)