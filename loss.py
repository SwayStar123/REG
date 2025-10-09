import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def _pairwise_sq_l2(z: torch.Tensor) -> torch.Tensor:
    """
    z: [N, D] float tensor
    returns: [N, N] matrix of squared L2 distances
    """
    # optional: compute in fp32 for stability, then cast back
    z32 = z if z.dtype == torch.float32 else z.float()
    n = (z32 * z32).sum(dim=1, keepdim=True)             # [N,1]
    d = n + n.t() - 2.0 * (z32 @ z32.t())                # [N,N]
    return d.clamp_min_(0.0)

def _logmeanexp(x: torch.Tensor, dim=None) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return (x - m).exp().mean(dim=dim).log() + m.squeeze(dim)

class DispersiveLoss(nn.Module):
    """
    InfoNCE-style dispersive loss without positives (Eq. (6) in paper), ℓ2 distance, no feature normalization.
    L_disp = log E_{i,j} [ exp( - ||z_i - z_j||^2 / tau ) ]
    If given a list of hidden states, computes per-layer and averages.
    """
    def __init__(self, tau: float = 0.5, lambda_weight: float = 0.5):
        super().__init__()
        self.tau = tau
        self.lambda_weight = lambda_weight

    @torch.autocast(device_type="cuda", enabled=False)
    def _disp_one(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [N, T, D] or [N, D] activation from one Transformer block
        flattens to [N, T*D] then computes dispersive loss
        """
        if h.dim() == 3:  # [N,T,D]
            h = h.reshape(h.shape[0], -1)
        # no normalization (paper’s best)
        d = _pairwise_sq_l2(h)                              # [N,N]
        v = (-d / self.tau).reshape(-1)                     # [N*N]
        return _logmeanexp(v, dim=0)                        # scalar

    def forward(self, hiddens) -> torch.Tensor:
        """
        hiddens: Tensor or List[Tensor] of per-block activations
        returns: lambda_weight * mean_layer_disp
        """
        if isinstance(hiddens, (list, tuple)):
            vals = [self._disp_one(h) for h in hiddens]
            disp = torch.stack(vals).mean()
        else:
            disp = self._disp_one(hiddens)
        return self.lambda_weight * disp

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.disp_crit = DispersiveLoss(tau=0.5, lambda_weight=0.5)

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None, cls_token=None,
                 time_input=None, noises=None,):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if time_input is None:
            if self.weighting == "uniform":
                time_input = torch.rand((images.shape[0], 1, 1, 1))
            elif self.weighting == "lognormal":
                # sample timestep according to log-normal distribution of sigmas following EDM
                rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
                sigma = rnd_normal.exp()
                if self.path_type == "linear":
                    time_input = sigma / (1 + sigma)
                elif self.path_type == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)

        if noises is None:
            noises = torch.randn_like(images)
            noises_cls = torch.randn_like(cls_token)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        cls_input = alpha_t.squeeze(-1).squeeze(-1) * cls_token + sigma_t.squeeze(-1).squeeze(-1) * noises_cls
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
            cls_target = d_alpha_t * cls_token + d_sigma_t * noises_cls
        else:
            raise NotImplementedError()

        model_output, zs_tilde, cls_output, hiddens = model(model_input, time_input.flatten(), **model_kwargs,
                                                    cls_token=cls_input, return_hiddens=True)

        disp_loss = self.disp_crit(hiddens)

        #denoising_loss
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_cls = mean_flat((cls_output - cls_target) ** 2)

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)


        return denoising_loss, proj_loss, time_input, noises, denoising_loss_cls, disp_loss
