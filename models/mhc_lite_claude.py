from __future__ import annotations
from typing import Callable
from functools import partial
from random import randrange

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten
import itertools

"""
mHC-lite: Fast implementation with minimal overhead.
"""

# ============================================================================
# Helper Functions
# ============================================================================

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def add(x, y):
    return x + y


# ============================================================================
# Main Functions
# ============================================================================

def get_expand_reduce_stream_functions(num_streams, dim=None, disable=False, **kwargs):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())
    
    expand_fn = ExpandStreams(num_streams)
    reduce_fn = ReduceStreams(num_streams)
    return expand_fn, reduce_fn


def get_init_and_expand_reduce_stream_functions(
    num_streams, num_fracs=1, dim=None, disable=None, **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)
    hyper_conn_klass = MHCLite if not disable else Residual
    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, dim=dim, disable=disable)
    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)
    return (init_hyper_conn_fn, *expand_reduce_fns)


class ExpandStreams(Module):
    def __init__(self, num_streams):
        super().__init__()
        self.s = num_streams
    
    def forward(self, x):
        # (B, ...) -> (B*s, ...)
        return x.repeat(self.s, *([1] * (x.dim() - 1)))


class ReduceStreams(Module):
    def __init__(self, num_streams):
        super().__init__()
        self.s = num_streams
    
    def forward(self, x):
        # (B*s, T, D) -> (B, T, D)
        B_s = x.shape[0]
        B = B_s // self.s
        return x.view(self.s, B, *x.shape[1:]).sum(0)


# ============================================================================
# Residual Base Class
# ============================================================================

class Residual(Module):
    def __init__(self, *args, branch=None, **kwargs):
        super().__init__()
        self.branch = branch

    def forward(self, residuals, *args, **kwargs):
        def add_residual_fn(out):
            (out, *rest), spec = tree_flatten(out)
            return tree_unflatten((out + residuals, *rest), spec)
        if not exists(self.branch):
            return residuals, add_residual_fn
        return add_residual_fn(self.branch(residuals, *args, **kwargs))


# ============================================================================
# MHCLite - Optimized Implementation
# ============================================================================

class MHCLite(Module):
    """
    Fast mHC-lite using convex combination of permutation matrices.
    Optimized to minimize memory operations and maximize fusion.
    """
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch=None,
        layer_index=None,
        dropout=0.,
        add_branch_out_to_residual=True,
        num_fracs=1,
        # Unused but kept for API compatibility
        channel_first=False,
        residual_transform=None,
        num_input_views=1,
        depth_residual_fn=add,
    ):
        super().__init__()
        assert num_fracs == 1, "This optimized version only supports num_fracs=1"
        assert num_input_views == 1, "This optimized version only supports num_input_views=1"
        
        self.branch = branch
        self.s = num_residual_streams
        self.dim = dim
        self.add_branch_out_to_residual = add_branch_out_to_residual
        
        # Compute n!
        n_fact = 1
        for i in range(2, num_residual_streams + 1):
            n_fact *= i
        self.n_fact = n_fact
        
        # Pre-compute and register permutation matrices as buffer
        perms = list(itertools.permutations(range(num_residual_streams)))
        perm_idx = torch.tensor(perms, dtype=torch.long)
        eye = torch.eye(num_residual_streams)
        perm_mats = eye[perm_idx]  # (n!, s, s)
        self.register_buffer('perm_mats', perm_mats, persistent=False)
        
        # Initialization index
        init_idx = default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        
        # Combined projection for alpha_pre (s values) and alpha_res (n! values)
        # Input: (B, T, s*d) after flattening streams
        # Output: (B, T, s + n!) for pre and res weights
        self.alpha_proj = nn.Linear(dim * num_residual_streams, num_residual_streams + n_fact, bias=True)
        
        # Initialize bias for identity-like behavior at init
        with torch.no_grad():
            # Pre weights: all -1 except init_idx = 1
            self.alpha_proj.bias[:num_residual_streams].fill_(-1.0)
            self.alpha_proj.bias[init_idx] = 1.0
            # Res weights: all -8 except identity perm = 0
            self.alpha_proj.bias[num_residual_streams:].fill_(-8.0)
            self.alpha_proj.bias[num_residual_streams] = 0.0  # Identity perm is first
            # Small init for weights
            self.alpha_proj.weight.mul_(0.01)
        
        # Beta projection for depth connection
        if add_branch_out_to_residual:
            self.beta_proj = nn.Linear(dim * num_residual_streams, num_residual_streams, bias=True)
            with torch.no_grad():
                self.beta_proj.bias.fill_(-1.0)
                self.beta_proj.bias[init_idx] = 1.0
                self.beta_proj.weight.mul_(0.01)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # RMSNorm scale
        self.norm_scale = (dim * num_residual_streams) ** 0.5

    def forward(self, residuals, *args, **kwargs):
        """
        residuals: (B*s, T, d)
        Returns: branch_input (B, T, d), add_residual_fn
        """
        s = self.s
        B_s, T, d = residuals.shape
        B = B_s // s
        
        # Reshape: (B*s, T, d) -> (B, T, s, d)
        x = residuals.view(B, s, T, d).transpose(1, 2)  # (B, T, s, d)
        
        # Flatten streams for projection: (B, T, s*d)
        x_flat = x.reshape(B, T, s * d)
        
        # Fast RMSNorm (no learnable params needed here)
        x_norm = F.normalize(x_flat.float(), dim=-1) * self.norm_scale
        
        # Project to get alpha weights: (B, T, s + n!)
        alpha_all = self.alpha_proj(x_norm)  # (B, T, s + n!)
        
        # Split into pre and res
        alpha_pre_logits = alpha_all[..., :s]           # (B, T, s)
        alpha_res_logits = alpha_all[..., s:]           # (B, T, n!)
        
        # Pre: sigmoid for [0, 1] weights
        alpha_pre = torch.sigmoid(alpha_pre_logits)     # (B, T, s)
        
        # Res: softmax over permutations, then weighted sum
        alpha_res_weights = F.softmax(alpha_res_logits.float(), dim=-1)  # (B, T, n!)
        
        # Compute doubly stochastic matrix: (B, T, n!) @ (n!, s, s) -> (B, T, s, s)
        # Use einsum for efficiency
        H_res = torch.einsum('btn,nij->btij', alpha_res_weights, self.perm_mats)  # (B, T, s, s)
        
        # Width connection:
        # branch_input = sum_i alpha_pre[i] * x[i]  (aggregation)
        # new_residuals[j] = sum_i H_res[i,j] * x[i]  (mixing)
        
        # Branch input: weighted sum over streams -> (B, T, d)
        branch_input = torch.einsum('bts,btsd->btd', alpha_pre, x.to(alpha_pre.dtype))
        
        # New residuals: mix via doubly stochastic matrix -> (B, T, s, d)
        new_residuals = torch.einsum('btij,btid->btjd', H_res, x.to(H_res.dtype))
        
        # Compute beta for depth connection
        if self.add_branch_out_to_residual:
            beta_logits = self.beta_proj(x_norm)  # (B, T, s)
            beta = torch.sigmoid(beta_logits) * 2  # (B, T, s) in [0, 2]
        
        # Reshape residuals back: (B, T, s, d) -> (B*s, T, d)
        new_residuals = new_residuals.transpose(1, 2).reshape(B * s, T, d)
        
        # Closure for adding residuals after branch
        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            
            (branch_out, *rest), spec = tree_flatten(branch_out)
            # branch_out: (B, T, d)
            # Scale by beta and add to each stream
            # beta: (B, T, s), branch_out: (B, T, d) -> (B, T, s, d)
            scaled = torch.einsum('bts,btd->btsd', beta, branch_out)
            # Reshape to (B*s, T, d) and add
            scaled = scaled.transpose(1, 2).reshape(B * s, T, d)
            out = self.dropout(scaled + new_residuals)
            return tree_unflatten((out, *rest), spec)
        
        if not exists(self.branch):
            return branch_input.to(residuals.dtype), add_residual_fn
        
        branch_output = self.branch(branch_input.to(residuals.dtype), *args, **kwargs)
        return add_residual_fn(branch_output)


# Attach static methods
MHCLite.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
MHCLite.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)