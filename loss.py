import torch
import numpy as np
import torch.nn.functional as F
import math

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
            cfm_weighting="uniform",
            encoders=[], 
            accelerator=None, 
            apply_time_shift=False,
            shift_base=4096,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.cfm_weighting = cfm_weighting
        self.apply_time_shift = apply_time_shift
        self.shift_base = shift_base

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
        
        if self.apply_time_shift:
            shift_dim = images.shape[1] * images.shape[2] * images.shape[3]
            shift = math.sqrt(shift_dim / self.shift_base)
            time_input = (shift * time_input) / (1 + (shift - 1) * time_input)
            time_input = torch.clamp(time_input, 0.0, 1.0)

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

        model_output, zs_tilde, cls_output, proj_ids_keeps = model(
            model_input,
            time_input.flatten(),
            **model_kwargs,
            cls_token=cls_input,
            return_ids=True,
        )

        #denoising_loss
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_cls = mean_flat((cls_output - cls_target) ** 2)

        # projection loss: align teacher z with student outputs. For middle
        # blocks we use sparse student tokens and the corresponding ids_keep;
        # for encoder/decoder depths we match all tokens densely.
        # zs: list of [B, T_full, C] teacher projections
        # zs_tilde: list of [B, T_keep or T_full, C] student projections
        proj_loss = 0.0
        num_terms = 0
        for z, z_tilde, ids_keep in zip(zs, zs_tilde, proj_ids_keeps):
            # If no ids_keep was provided for this projection, compare all
            # tokens directly. Otherwise gather teacher tokens at the kept
            # sparse indices (ids_keep: [B, T_keep]).
            if ids_keep is None:
                z_sel = z
            else:
                B, T_full, C = z.shape
                z_sel = z.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, C))

            # z_sel and z_tilde should now both be [B, T_keep, C] (or [B, T_full, C] in dense case)
            z_sel = torch.nn.functional.normalize(z_sel, dim=-1)
            z_tilde_norm = torch.nn.functional.normalize(z_tilde, dim=-1)

            proj_loss += mean_flat(-(z_sel * z_tilde_norm).sum(dim=-1))
            num_terms += 1

        if num_terms > 0:
            proj_loss /= num_terms

        cfm_target = torch.roll(model_target, shifts=1, dims=0)
        cfm_target_cls = torch.roll(cls_target, shifts=1, dims=0)
        if self.cfm_weighting == "uniform":
            cfm_loss = -((model_output - cfm_target) ** 2).mean()
            cfm_loss_cls = -((cls_output - cfm_target_cls) ** 2).mean()
        elif self.cfm_weighting == "linear":
            cfm_loss = -(((model_output - cfm_target) ** 2) * time_input).mean()
            cfm_loss_cls = -(((cls_output - cfm_target_cls) ** 2) * time_input).mean()

        return denoising_loss, proj_loss, time_input, noises, denoising_loss_cls, cfm_loss, cfm_loss_cls
