from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange
import itertools
from functools import lru_cache

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Reduce

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def add(x, y):
    return x + y

# ---------------------------------------------------------------------
# permutation index cache (NO permutation matrices ever)
# perm_idx[r, i] = source stream index feeding output stream i under permutation r
# invperm[r, s] = output stream index i such that perm_idx[r, i] == s
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def _perm_idx_cpu(s: int) -> torch.Tensor:
    perms = list(itertools.permutations(range(s)))
    return torch.tensor(perms, dtype=torch.int32)  # (R, S)

@lru_cache(maxsize=None)
def _invperm_cpu(s: int) -> torch.Tensor:
    perm = _perm_idx_cpu(s)  # (R, S)
    R, S = perm.shape
    inv = torch.empty((R, S), dtype=torch.int32)
    for r in range(R):
        for i in range(S):
            inv[r, perm[r, i].item()] = i
    return inv

_perm_cache = {}   # (S, device_str) -> perm_idx (R,S)
_inv_cache = {}    # (S, device_str) -> invperm (R,S)

def get_perm_and_invperm(s: int, device: torch.device):
    key = (s, str(device))
    if key not in _perm_cache:
        _perm_cache[key] = _perm_idx_cpu(s).to(device=device, non_blocking=True)
        _inv_cache[key] = _invperm_cpu(s).to(device=device, non_blocking=True)
    return _perm_cache[key], _inv_cache[key]

# ---------------------------------------------------------------------
# Triton (forward + backward) with proper autograd
# y[b,i,d] = sum_r w[b,r] * x[b, perm[r,i], d]
# ---------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    @triton.jit
    def perm_mix_fwd_kernel(
        X_ptr, W_ptr, PERM_ptr, Y_ptr,
        B: tl.constexpr, S: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
        stride_xb: tl.constexpr, stride_xs: tl.constexpr, stride_xd: tl.constexpr,
        stride_wb: tl.constexpr, stride_wr: tl.constexpr,
        stride_pr: tl.constexpr, stride_ps: tl.constexpr,
        stride_yb: tl.constexpr, stride_ys: tl.constexpr, stride_yd: tl.constexpr,
        OUT_BF16: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)  # output stream i
        pid_d = tl.program_id(2)

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # unrolled small-R if S small (e.g. S=4 -> R=24)
        for r in tl.static_range(0, R):
            w = tl.load(W_ptr + pid_b * stride_wb + r * stride_wr).to(tl.float32)
            src = tl.load(PERM_ptr + r * stride_pr + pid_s * stride_ps).to(tl.int32)
            x_ptr = X_ptr + pid_b * stride_xb + src * stride_xs + d_off * stride_xd
            x = tl.load(x_ptr, mask=d_mask, other=0.0).to(tl.float32)
            acc += w * x

        y_ptr = Y_ptr + pid_b * stride_yb + pid_s * stride_ys + d_off * stride_yd
        if OUT_BF16:
            tl.store(y_ptr, acc.to(tl.bfloat16), mask=d_mask)
        else:
            tl.store(y_ptr, acc.to(tl.float16), mask=d_mask)

    @triton.jit
    def perm_mix_bwd_x_kernel(
        GO_ptr, W_ptr, INVPERM_ptr, GX_ptr,
        B: tl.constexpr, S: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
        stride_gob: tl.constexpr, stride_gos: tl.constexpr, stride_god: tl.constexpr,
        stride_wb: tl.constexpr, stride_wr: tl.constexpr,
        stride_ir: tl.constexpr, stride_is: tl.constexpr,
        stride_gxb: tl.constexpr, stride_gxs: tl.constexpr, stride_gxd: tl.constexpr,
        OUT_BF16: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_src = tl.program_id(1)  # source stream s
        pid_d = tl.program_id(2)

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # gx[b,src,d] = sum_r w[b,r] * go[b, invperm[r,src], d]
        for r in tl.static_range(0, R):
            w = tl.load(W_ptr + pid_b * stride_wb + r * stride_wr).to(tl.float32)
            out_i = tl.load(INVPERM_ptr + r * stride_ir + pid_src * stride_is).to(tl.int32)
            go_ptr = GO_ptr + pid_b * stride_gob + out_i * stride_gos + d_off * stride_god
            go = tl.load(go_ptr, mask=d_mask, other=0.0).to(tl.float32)
            acc += w * go

        gx_ptr = GX_ptr + pid_b * stride_gxb + pid_src * stride_gxs + d_off * stride_gxd
        if OUT_BF16:
            tl.store(gx_ptr, acc.to(tl.bfloat16), mask=d_mask)
        else:
            tl.store(gx_ptr, acc.to(tl.float16), mask=d_mask)

    @triton.jit
    def perm_mix_bwd_w_kernel(
        X_ptr, GO_ptr, PERM_ptr, GW_ptr,
        B: tl.constexpr, S: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
        stride_xb: tl.constexpr, stride_xs: tl.constexpr, stride_xd: tl.constexpr,
        stride_gob: tl.constexpr, stride_gos: tl.constexpr, stride_god: tl.constexpr,
        stride_pr: tl.constexpr, stride_ps: tl.constexpr,
        stride_gwb: tl.constexpr, stride_gwr: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ND: tl.constexpr,          # <-- number of blocks over D (compile-time)
    ):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)

        acc = tl.zeros((), dtype=tl.float32)

        # gw[b,r] = sum_{i,d} go[b,i,d] * x[b, perm[r,i], d]
        for i in tl.static_range(0, S):
            src = tl.load(PERM_ptr + pid_r * stride_pr + i * stride_ps).to(tl.int32)

            for d0 in tl.static_range(0, ND):
                d_off = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
                mask = d_off < D

                x_ptr = X_ptr + pid_b * stride_xb + src * stride_xs + d_off * stride_xd
                go_ptr = GO_ptr + pid_b * stride_gob + i * stride_gos + d_off * stride_god

                x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
                go = tl.load(go_ptr, mask=mask, other=0.0).to(tl.float32)

                acc += tl.sum(x * go, axis=0)

        tl.store(GW_ptr + pid_b * stride_gwb + pid_r * stride_gwr, acc)


class _PermMixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bsd: torch.Tensor, w_br: torch.Tensor, perm_idx_rs: torch.Tensor, invperm_rs: torch.Tensor):
        """
        x_bsd: (B, S, D) fp16/bf16
        w_br:  (B, R)    fp16/bf16/fp32 (typically softmax output)
        perm_idx_rs: (R, S) int32
        invperm_rs:  (R, S) int32
        """
        if (not TRITON_AVAILABLE) or (not x_bsd.is_cuda):
            # differentiable fallback
            gathered = x_bsd[:, perm_idx_rs]                      # (B, R, S, D)
            y = (w_br[:, :, None, None] * gathered).sum(dim=1)    # (B, S, D)
            ctx.save_for_backward(x_bsd, w_br, perm_idx_rs, invperm_rs)
            ctx.use_triton = False
            return y

        B, S, D = x_bsd.shape
        R = w_br.shape[1]

        y = torch.empty((B, S, D), device=x_bsd.device, dtype=x_bsd.dtype)

        out_bf16 = (x_bsd.dtype == torch.bfloat16)
        BLOCK_D = 128
        grid = (B, S, triton.cdiv(D, BLOCK_D))
        perm_mix_fwd_kernel[grid](
            x_bsd, w_br, perm_idx_rs, y,
            B=B, S=S, D=D, R=R,
            stride_xb=x_bsd.stride(0), stride_xs=x_bsd.stride(1), stride_xd=x_bsd.stride(2),
            stride_wb=w_br.stride(0),  stride_wr=w_br.stride(1),
            stride_pr=perm_idx_rs.stride(0), stride_ps=perm_idx_rs.stride(1),
            stride_yb=y.stride(0), stride_ys=y.stride(1), stride_yd=y.stride(2),
            OUT_BF16=out_bf16,
            BLOCK_D=BLOCK_D,
            num_warps=4 if D >= 256 else 2,
        )
        ctx.save_for_backward(x_bsd, w_br, perm_idx_rs, invperm_rs)
        ctx.use_triton = True
        ctx.block_d = BLOCK_D
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_bsd, w_br, perm_idx_rs, invperm_rs = ctx.saved_tensors

        # fallback path
        if not getattr(ctx, "use_triton", False) or (not TRITON_AVAILABLE) or (not grad_out.is_cuda):
            # grad_x
            # gx[b, src, d] = sum_r w[b,r] * go[b, invperm[r,src], d]
            gx = torch.zeros_like(x_bsd)
            B, S, D = x_bsd.shape
            R = w_br.shape[1]
            for r in range(R):
                out_i = invperm_rs[r]  # (S,)
                go_r = grad_out[:, out_i, :]  # (B,S,D)
                gx += w_br[:, r].view(B, 1, 1) * go_r

            # grad_w
            # gw[b,r] = sum_{i,d} go[b,i,d] * x[b, perm[r,i], d]
            gathered = x_bsd[:, perm_idx_rs]  # (B,R,S,D)
            gw = (grad_out[:, None, :, :] * gathered).sum(dim=(2, 3))  # (B,R)

            return gx, gw, None, None

        # Triton backward kernels
        B, S, D = x_bsd.shape
        R = w_br.shape[1]

        gx = torch.empty_like(x_bsd, dtype=x_bsd.dtype)
        # gw is float32 (best for stability); it will backprop through softmax fine
        gw = torch.empty((B, R), device=x_bsd.device, dtype=torch.float32)

        out_bf16 = (x_bsd.dtype == torch.bfloat16)
        BLOCK_D = getattr(ctx, "block_d", 128)

        # grad_x
        grid_x = (B, S, triton.cdiv(D, BLOCK_D))
        perm_mix_bwd_x_kernel[grid_x](
            grad_out, w_br, invperm_rs, gx,
            B=B, S=S, D=D, R=R,
            stride_gob=grad_out.stride(0), stride_gos=grad_out.stride(1), stride_god=grad_out.stride(2),
            stride_wb=w_br.stride(0),      stride_wr=w_br.stride(1),
            stride_ir=invperm_rs.stride(0), stride_is=invperm_rs.stride(1),
            stride_gxb=gx.stride(0), stride_gxs=gx.stride(1), stride_gxd=gx.stride(2),
            OUT_BF16=out_bf16,
            BLOCK_D=BLOCK_D,
            num_warps=4 if D >= 256 else 2,
        )

        # grad_w
        grid_w = (B, R)

        # compile-time constant number of D blocks
        ND = (D + BLOCK_D - 1) // BLOCK_D

        perm_mix_bwd_w_kernel[grid_w](
            x_bsd, grad_out, perm_idx_rs, gw,
            B=B, S=S, D=D, R=R,
            stride_xb=x_bsd.stride(0), stride_xs=x_bsd.stride(1), stride_xd=x_bsd.stride(2),
            stride_gob=grad_out.stride(0), stride_gos=grad_out.stride(1), stride_god=grad_out.stride(2),
            stride_pr=perm_idx_rs.stride(0), stride_ps=perm_idx_rs.stride(1),
            stride_gwb=gw.stride(0), stride_gwr=gw.stride(1),
            BLOCK_D=BLOCK_D,
            ND=ND,                 # <-- new
            num_warps=4,
        )

        # cast grad_w back to w dtype if needed
        if w_br.dtype != torch.float32:
            gw = gw.to(w_br.dtype)

        return gx, gw, None, None

def perm_mix(x_bsd: torch.Tensor, w_br: torch.Tensor, perm_idx_rs: torch.Tensor, invperm_rs: torch.Tensor):
    return _PermMixFn.apply(x_bsd, w_br, perm_idx_rs, invperm_rs)

# ---------------------------------------------------------------------
# stream expand/reduce helpers
# ---------------------------------------------------------------------

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed=False,
    dim=None,
    disable=False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), "`dim` must be passed if add_stream_embed=True"
        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        expand_fn = Reduce(pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams)

    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)
    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)
    hyper_conn_klass = MHCLite if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# ---------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------

class RMSNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

# ---------------------------------------------------------------------
# residual base class
# ---------------------------------------------------------------------

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)

# ---------------------------------------------------------------------
# MHCLite (optimized: no permutation matrices + Triton autograd)
# ---------------------------------------------------------------------

class MHCLite(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index=None,
        channel_first=False,
        dropout=0.0,
        residual_transform: Module | None = None,
        add_branch_out_to_residual=True,
        num_input_views=1,
        depth_residual_fn=add,
        num_fracs=1,
    ):
        super().__init__()
        self.branch = branch

        assert num_fracs >= 1
        assert num_residual_streams > 0
        assert num_input_views >= 1
        assert divisible_by(dim, num_fracs), f"feature dimension ({dim}) must be divisible by num_fracs ({num_fracs})"

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        # fraction split/merge
        self.split_fracs = lambda x: x.view(*x.shape[:-1], num_fracs, dim // num_fracs)
        self.merge_fracs = lambda x: x.reshape(*x.shape[:-2], dim)

        # effective dim per fraction
        d_per = dim // num_fracs

        self.num_residual_streams = num_residual_streams
        self.num_input_views = num_input_views

        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams

        # number of perms
        self.num_perms = len(list(itertools.permutations(range(num_residual_streams))))

        # norm over concatenated (streams * d_per) PER FRACTION (fused by flattening)
        self.norm = RMSNorm(d_per * num_residual_streams)

        # --- parameters (kept simple, per-frac; avoids dead params / DDP unused) ---
        # dynamic alpha: for each fraction, map (S*d_per) -> (S*V + R)
        out_alpha = num_residual_streams * num_input_views + self.num_perms
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(num_fracs, d_per * num_residual_streams, out_alpha))

        # static alpha: per fraction
        static_alpha = torch.zeros(num_fracs, out_alpha)
        # init alpha_pre: favor one residual stream into all views
        for f in range(num_fracs):
            pre = torch.full((num_residual_streams * num_input_views,), -1.0)
            for v in range(num_input_views):
                pre[init_residual_index * num_input_views + v] = 1.0
            static_alpha[f, :num_residual_streams * num_input_views] = pre
            # init perm weights: favor identity perm (index 0)
            perm = torch.full((self.num_perms,), -8.0)
            perm[0] = 0.0
            static_alpha[f, num_residual_streams * num_input_views:] = perm
        self.static_alpha = nn.Parameter(static_alpha)

        self.pre_branch_scale = nn.Parameter(torch.ones(()) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # depth / beta
        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(num_fracs, d_per * num_residual_streams, num_residual_streams))
            static_beta = torch.full((num_fracs, num_residual_streams), -1.0)
            static_beta[:, init_residual_index] = 1.0
            self.static_beta = nn.Parameter(static_beta)
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)
        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals: torch.Tensor):
        S = self.num_residual_streams
        Ff = self.num_fracs
        V = self.num_input_views
        R = self.num_perms

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        assert residuals.shape[0] % S == 0
        b = residuals.shape[0] // S
        *spatial, dim = residuals.shape[1:]

        # split fractions
        x = self.split_fracs(residuals)  # (b*S, ..., F, d_per)

        # reshape streams: (b, ..., F, S, d_per)
        x = rearrange(x, "(b s) ... f d -> b ... f s d", s=S)

        # flatten tokens for speed: T = b * prod(spatial)
        if len(spatial) > 0:
            x = x.reshape(b, -1, Ff, S, x.shape[-1])  # (b, T, F, S, d_per)
            x = x.reshape(-1, Ff, S, x.shape[-1])     # (Ttot, F, S, d_per)
        else:
            x = x.reshape(b, Ff, S, x.shape[-1])      # (b, F, S, d_per)

        Ttot = x.shape[0]
        d_per = x.shape[-1]

        # norm per token per frac over (S*d_per)
        x_flat = x.reshape(Ttot, Ff, S * d_per)
        x_norm = self.norm(x_flat)  # (Ttot, F, S*d_per)

        # dynamic alpha
        # (Ttot, F, K) @ (F, K, out_alpha) -> (Ttot, F, out_alpha)
        wc = torch.einsum("t f k, f k o -> t f o", x_norm, self.dynamic_alpha_fn)
        wc = wc + self.static_alpha  # broadcast (F, out_alpha)

        # split into pre + perm
        pre_logits = wc[:, :, :S * V] * self.pre_branch_scale               # (Ttot, F, S*V)
        perm_logits = wc[:, :, S * V:] * self.residual_scale                # (Ttot, F, R)

        alpha_pre = torch.sigmoid(pre_logits).view(Ttot, Ff, S, V)          # (Ttot,F,S,V)
        w_perm = torch.softmax(perm_logits, dim=-1)                         # (Ttot,F,R)

        # branch input: (Ttot,F,V,d_per) = sum_s alpha_pre * x
        # bmm per frac:
        x_f = x.view(Ttot * Ff, S, d_per)
        w_pre = alpha_pre.permute(0, 1, 3, 2).contiguous().view(Ttot * Ff, V, S)
        branch_f = torch.bmm(w_pre, x_f).view(Ttot, Ff, V, d_per)

        # residual stream mixing via permutation mixture:
        perm_idx, invperm = get_perm_and_invperm(S, x.device)
        w_2d = w_perm.contiguous().view(Ttot * Ff, R)
        mixed_f = perm_mix(x_f, w_2d, perm_idx, invperm).view(Ttot, Ff, S, d_per)

        # beta (same-frac only; avoids odd cross-frac shapes and dead params)
        beta = None
        if self.add_branch_out_to_residual:
            dc = torch.einsum("t f k, f k s -> t f s", x_norm, self.dynamic_beta_fn)
            beta = dc * self.h_post_scale + self.static_beta
            beta = torch.sigmoid(beta) * 2.0  # (Ttot, F, S)

        # merge fracs back
        # branch: (Ttot, V, dim)
        branch = branch_f.permute(0, 2, 1, 3).contiguous().view(Ttot, V, Ff * d_per)
        # residuals: (Ttot, S, dim)
        mixed = mixed_f.permute(0, 2, 1, 3).contiguous().view(Ttot, S, Ff * d_per)

        # unflatten tokens back to (b, spatial..., ...)
        if len(spatial) > 0:
            T = int(torch.tensor(spatial).prod().item())
            branch = branch.view(b, T, V, dim).view(b, *spatial, V, dim)
            mixed = mixed.view(b, T, S, dim).view(b, *spatial, S, dim)

            # residuals_out: (b*S, spatial..., dim)
            residuals_out = mixed.permute(0, 1 + len(spatial), *range(1, 1 + len(spatial)), 2 + len(spatial)).contiguous()
            residuals_out = residuals_out.view(b * S, *spatial, dim)

            if V == 1:
                branch_input = branch[..., 0, :]
            else:
                branch_input = branch.permute(2, 0, *range(1, 1 + len(spatial)), 3).contiguous()
        else:
            residuals_out = mixed.view(b * S, dim)
            if V == 1:
                branch_input = branch[:, 0, :]
            else:
                branch_input = branch.permute(1, 0, 2).contiguous()

        if self.channel_first:
            residuals_out = rearrange(residuals_out, "b ... d -> b d ...")
            if V == 1:
                branch_input = rearrange(branch_input, "b ... d -> b d ...")
            else:
                branch_input = rearrange(branch_input, "v b ... d -> v b d ...")

        return branch_input, residuals_out, dict(beta=beta, token_shape=(b, spatial, dim))

    def depth_connection(self, branch_output, residuals, *, beta, token_shape):
        """
        beta: (Ttot, F, S) where Ttot = b * prod(spatial)
        branch_output: (b, spatial..., dim) or (b, dim, spatial...) if channel_first
        residuals: (b*S, spatial..., dim) or (b*S, dim, spatial...) if channel_first
        """
        assert self.add_branch_out_to_residual

        S = self.num_residual_streams
        Ff = self.num_fracs

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")
            residuals = rearrange(residuals, "b d ... -> b ... d")

        b, spatial, dim = token_shape
        *spatial2, dim2 = branch_output.shape[1:]
        assert dim2 == dim

        # split fracs on branch output: (b, spatial..., F, d_per)
        bo = self.split_fracs(branch_output)

        # flatten tokens
        if len(spatial) > 0:
            Ttot = b * int(torch.tensor(spatial).prod().item())
            bo = bo.view(Ttot, Ff, bo.shape[-1])  # (Ttot, F, d_per)
        else:
            Ttot = b
            bo = bo.view(Ttot, Ff, bo.shape[-1])

        # out: (Ttot, F, S, d_per) = bo[:, :, None, :] * beta[:, :, :, None]
        out = bo[:, :, None, :] * beta[:, :, :, None]

        # merge fracs: (Ttot, S, dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(Ttot, S, dim)

        # unflatten back to (b*S, spatial..., dim)
        if len(spatial) > 0:
            out = out.view(b, -1, S, dim).view(b, *spatial, S, dim)
            out = out.permute(0, 1 + len(spatial), *range(1, 1 + len(spatial)), 2 + len(spatial)).contiguous()
            out = out.view(b * S, *spatial, dim)
        else:
            out = out.view(b * S, dim)

        # channel-first restore
        if self.channel_first:
            out = rearrange(out, "b ... d -> b d ...")
            residuals = rearrange(residuals, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(out, residuals)
        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals_out, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals_out, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)

MHCLite.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
MHCLite.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

# ---------------------------------------------------------------------
# StreamEmbed
# ---------------------------------------------------------------------

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first=False,
        expand_to_streams=False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams
        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):
        if self.expand_to_streams:
            residuals = repeat(residuals, "b ... -> (b s) ...", s=self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, "(b s) d ... -> b ... s d", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, "b ... s d -> (b s) d ...", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "b ... s d -> (b s) ... d", s=self.num_streams)

        return residuals

# ---------------------------------------------------------------------
# AttentionPoolReduceStream
# ---------------------------------------------------------------------

class AttentionPoolReduceStream(Module):
    def __init__(self, num_streams, dim, channel_first=False):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias=False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):
        if self.channel_first:
            residuals = rearrange(residuals, "(b s) d ... -> b ... s d", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim=-2)
        residuals = reduce(residuals * attn, "b ... s d -> b ... d", "sum")

        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")

        return residuals
