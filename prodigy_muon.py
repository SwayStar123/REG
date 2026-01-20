import math
import torch
import torch.distributed as dist
from torch import Tensor
from typing import TYPE_CHECKING, Any, Optional, List

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A 
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class ProdigyCombined(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: Optional[float] = None,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float('inf'),
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        rank: int = 0,
        world_size: int = 1,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
        fsdp_in_use: bool = False,
        slice_p: int = 1
    ):
        defaults = dict(
            lr=lr, betas=betas, beta3=beta3, eps=eps, 
            weight_decay=weight_decay,
            d=d0, d0=d0, d_max=d0, d_numerator=0.0, d_coef=d_coef,
            k=0, growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple, safeguard_warmup=safeguard_warmup,
            fsdp_in_use=fsdp_in_use, slice_p=slice_p,
            muon_momentum=muon_momentum,
            muon_ns_steps=muon_ns_steps,
            rank=rank,
            world_size=world_size
        )
        
        all_params = list(params)
        muon_params = []
        adam_params = []
        
        for p in all_params:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
                
        param_groups = [
            {'params': adam_params, 'type': 'adam'},
            {'params': muon_params, 'type': 'muon'}
        ]
        
        super().__init__(param_groups, defaults)
        self.d0 = d0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        d_denom = 0.0
        delta_numerator = 0.0
        
        group0 = self.param_groups[0]
        d = group0['d']
        d0 = group0['d0']
        
        # ---------------------------------------------------------------------
        # PHASE 1: Fast D-Calculation (Foreach)
        # ---------------------------------------------------------------------
        # We batch operations to avoid Python loop overhead
        
        for group in self.param_groups:
            beta3 = group['beta3']
            if beta3 is None: beta3 = math.sqrt(group['betas'][1])
            dlr = d * group['lr'] 
            slice_p = group['slice_p']
            safeguard = group['safeguard_warmup']
            
            # Lists for foreach ops
            grads_sliced = []
            p_sliced = []
            p0_sliced = []
            s_list = []
            
            # 1. Prepare Data (Lazy Init + Slicing)
            for p in group['params']:
                if p.grad is None: continue
                
                state = self.state[p]
                
                # Lazy Init
                if 'step' not in state:
                    state['step'] = 0
                    # Store only sliced p0
                    state['p0'] = p.detach().flatten()[::slice_p].clone()
                    state['s'] = torch.zeros_like(state['p0']) 
                    
                    if group['type'] == 'adam':
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    elif group['type'] == 'muon':
                        state['muon_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Collect Slices
                grads_sliced.append(p.grad.flatten()[::slice_p])
                p_sliced.append(p.flatten()[::slice_p])
                p0_sliced.append(state['p0'])
                s_list.append(state['s'])
            
            if len(grads_sliced) == 0: continue

            # 2. Batch Compute Numerator: sum( <g, p0 - p> )
            # diff = p0 - p
            diffs = torch._foreach_sub(p0_sliced, p_sliced)
            # prod = g * diff
            prods = torch._foreach_mul(grads_sliced, diffs)
            # sum all elements
            delta_numerator += (d / d0) * dlr * sum(t.sum().item() for t in prods)

            # 3. Batch Update 's' (running sum)
            # s = s * beta3 + grad * alpha
            alpha = (d / d0) * d if safeguard else (d / d0) * dlr
            torch._foreach_add_(s_list, grads_sliced, alpha=alpha)
            torch._foreach_mul_(s_list, beta3) # Note: Logic slightly rearranged for foreach compat, effectively same
            
            # sum abs(s)
            # There is no foreach_abs_sum, so we do quick loop or cat
            # Since s_list are sliced (small), this loop is fast enough
            for s in s_list:
                d_denom += s.abs().sum().item()

        # ---------------------------------------------------------------------
        # PHASE 2: Sync 'd'
        # ---------------------------------------------------------------------
        if d_denom == 0: return loss
        
        if self.param_groups[0]['fsdp_in_use'] or self.param_groups[0]['world_size'] > 1:
            dist_tensor = torch.tensor([delta_numerator, d_denom], device='cuda')
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
            global_d_numerator = group0['d_numerator'] + dist_tensor[0]
            global_d_denom = dist_tensor[1]
        else:
            global_d_numerator = group0['d_numerator'] + delta_numerator
            global_d_denom = d_denom

        d_hat = group0['d_coef'] * global_d_numerator / global_d_denom
        d_max = max(group0['d_max'], d_hat)
        d = min(d_max, max(d, d_hat))
        if group0['growth_rate'] != float('inf'):
            d = min(d, group0['d'] * group0['growth_rate'])
        
        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d'] = d
            group['d_max'] = d_max

        # ---------------------------------------------------------------------
        # PHASE 3: Fast Updates (Foreach for Adam, Loop for Muon)
        # ---------------------------------------------------------------------
        for group in self.param_groups:
            lr = group['lr']
            dlr = d * lr
            decay = group['weight_decay']
            decouple = group['decouple']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            # --- ADAM BRANCH (FUSED) ---
            if group['type'] == 'adam':
                # Collect lists for foreach
                params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                
                for p in group['params']:
                    if p.grad is not None:
                        params.append(p)
                        grads.append(p.grad)
                        state = self.state[p]
                        state['step'] += 1
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                
                if len(params) == 0: continue

                # Decoupled Weight Decay
                if decay != 0 and decouple:
                    torch._foreach_add_(params, params, alpha=-decay * dlr)
                elif decay != 0 and not decouple:
                    torch._foreach_add_(grads, params, alpha=decay)

                # Adam Steps
                # exp_avg = exp_avg * beta1 + grad * d * (1 - beta1)
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=d * (1 - beta1))

                # exp_avg_sq = exp_avg_sq * beta2 + grad^2 * d^2 * (1 - beta2)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=d * d * (1 - beta2))

                # denom = sqrt(exp_avg_sq) + d * eps
                # We calculate this into a temporary list
                denoms = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_add_(denoms, d * eps)

                # p = p - dlr * exp_avg / denom
                torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-dlr)

            # --- MUON BRANCH (LOOP) ---
            # Muon operates on large matrices, so loop overhead is negligible compared to matmul.
            # Foreach is hard here due to complex SVD-like logic.
            elif group['type'] == 'muon':
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad
                    state = self.state[p]
                    state['step'] += 1
                    
                    # Momentum
                    buf = state['muon_momentum']
                    buf.lerp_(grad, 1 - group['muon_momentum'])
                    g_to_ortho = grad.lerp(buf, group['muon_momentum'])
                    
                    # Newton-Schulz
                    if g_to_ortho.ndim == 4:
                        g_to_ortho = g_to_ortho.view(len(g_to_ortho), -1)
                        
                    update = zeropower_via_newtonschulz5(g_to_ortho, steps=group['muon_ns_steps'])
                    
                    if update.ndim == 2 and p.ndim == 4:
                        update = update.view_as(p)
                        
                    muon_scaling = max(1, p.size(-2) / p.size(-1))**0.5
                    
                    # Update
                    p.data.mul_(1 - dlr * decay)
                    p.data.add_(update, alpha=-dlr * muon_scaling)

        return loss