import torch
import numpy as np
import torch.nn.functional as F

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
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            c_type="truncated",
            a=0.8,
            b=1.0,
            lambd=4.0,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.encoders = encoders
        self.accelerator = accelerator
        self.c_type = c_type
        self.a = a
        self.b = b
        self.lambd = lambd

    def interpolant(self, t):
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = -1
        d_sigma_t =  1

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def c_gamma(self, gamma):
        if self.c_type == "linear":
            return self.lambd * (1 - gamma)
        elif self.c_type == "truncated":
            c = torch.where(
                gamma <= self.a,
                torch.ones_like(gamma),
                (1 - gamma) / (1 - self.a)
            )
            return self.lambd * c
        elif self.c_type == "piecewise":
            c = torch.where(
                gamma <= self.a,
                self.b - (self.b - 1) / self.a * gamma,
                (1 - gamma) / (1 - self.a)
            )
            return self.lambd * c
        elif self.c_type == "constant":
            return self.lambd * torch.ones_like(gamma)
        else:
            raise NotImplementedError(f"Unknown c_type: {self.c_type}")

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
                time_input = sigma / (1 + sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        if noises is None:
            noises = torch.randn_like(images)
            noises_cls = torch.randn_like(cls_token)

        if self.prediction == "gradient":
            gamma = time_input
            x_gamma = gamma * images + (1 - gamma) * noises
            cls_gamma = gamma.squeeze(-1).squeeze(-1) * cls_token + (1 - gamma.squeeze(-1).squeeze(-1)) * noises_cls

            c_val = self.c_gamma(gamma)
            c_val_cls = self.c_gamma(gamma.squeeze(-1).squeeze(-1))

            model_target = (noises - images) * c_val
            cls_target = (noises_cls - cls_token) * c_val_cls

            dummy_time = torch.zeros(images.shape[0], device=images.device, dtype=images.dtype)
            model_output, zs_tilde, cls_output = model(x_gamma, dummy_time, **model_kwargs,
                                                       cls_token=cls_gamma)
        else:
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

            model_input = alpha_t * images + sigma_t * noises
            cls_input = alpha_t.squeeze(-1).squeeze(-1) * cls_token + sigma_t.squeeze(-1).squeeze(-1) * noises_cls
            if self.prediction == 'v':
                model_target = d_alpha_t * images + d_sigma_t * noises
                cls_target = d_alpha_t * cls_token + d_sigma_t * noises_cls
            elif self.prediction == "x":
                model_target = (model_input - images) / (sigma_t + 0.05)
                cls_target = (cls_input - cls_token) / (sigma_t + 0.05)

            else:
                raise NotImplementedError()

            model_output, zs_tilde, cls_output = model(model_input, time_input.flatten(), **model_kwargs,
                                                        cls_token=cls_input)

            if self.prediction == "x":
                model_output = (model_input - model_output) / (sigma_t + 0.05)
                cls_output = (cls_input - cls_output) / (sigma_t + 0.05)
        


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

        return denoising_loss, proj_loss, time_input, noises, denoising_loss_cls
