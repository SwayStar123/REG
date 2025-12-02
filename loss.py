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

def _drop_cls(x):  # x: [B, T, D], drop token 0
    return x[:, 1:, :]

def _autocorr(tokens_no_cls: torch.Tensor) -> torch.Tensor:
    # tokens_no_cls: [B, N, D]; cosine autocorr A(H) = Hn Hn^T
    Hn = F.normalize(tokens_no_cls, dim=-1)
    return torch.einsum('bid,bjd->bij', Hn, Hn)  # [B, N, N]

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            ema_model=None,
            guidance_mode="none",
            max_w=6.0,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.ema_model = ema_model
        self.guidance_mode = guidance_mode
        self.max_w = max_w

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
            base_target = d_alpha_t * images + d_sigma_t * noises
            base_target_cls = d_alpha_t * cls_token + d_sigma_t * noises_cls
        else:
            raise NotImplementedError()

        # sample random guidance weight w in [0, max_w]
        bsz = images.shape[0]
        w = torch.rand(bsz, device=images.device, dtype=images.dtype) * self.max_w

        # forward student with optional conditioning on w
        student_kwargs = dict(model_kwargs)
        student_kwargs["w"] = w
        model_output, zs_tilde, cls_output = model(model_input, time_input.flatten(), **student_kwargs,
                                                    cls_token=cls_input)

        # default targets are the base stochastic interpolant targets (ground-truth flow v)
        model_target = base_target
        cls_target = base_target_cls

        # EMA-based guided distillation targets
        if self.guidance_mode != "none" and self.ema_model is not None and model_kwargs is not None and "y" in model_kwargs:
            labels = model_kwargs["y"]
            num_classes = getattr(self.ema_model, "num_classes", None)
            if num_classes is None:
                raise ValueError("EMA model must have num_classes attribute for guided distillation.")

            w_img = w.view(-1, 1, 1, 1)
            w_cls = w.view(-1, 1)

            # model guidance: base_target + w * (ema_cond - ema_uncond)
            if self.guidance_mode == "model":
                with torch.no_grad():
                    y_cond = labels
                    y_uncond = torch.full_like(labels, num_classes, device=labels.device)

                    ema_input = torch.cat([model_input, model_input], dim=0)
                    ema_cls_input = torch.cat([cls_input, cls_input], dim=0)
                    ema_t = torch.cat([time_input.flatten(), time_input.flatten()], dim=0)
                    ema_y = torch.cat([y_cond, y_uncond], dim=0)

                    ema_out, _, ema_cls_out = self.ema_model(ema_input, ema_t, ema_y, cls_token=ema_cls_input)
                    ema_cond, ema_uncond = ema_out.chunk(2)
                    ema_cls_cond, ema_cls_uncond = ema_cls_out.chunk(2)

                model_target = base_target + w_img * (ema_cond - ema_uncond)
                cls_target = base_target_cls + w_cls * (ema_cls_cond - ema_cls_uncond)

            # truth guidance: base_target + w * (base_target - ema_uncond)
            elif self.guidance_mode == "truth":
                with torch.no_grad():
                    y_uncond = torch.full_like(labels, num_classes, device=labels.device)

                    ema_input = model_input
                    ema_cls_input = cls_input
                    ema_t = time_input.flatten()
                    ema_y = y_uncond

                    ema_uncond, _, ema_cls_uncond = self.ema_model(ema_input, ema_t, ema_y, cls_token=ema_cls_input)

                model_target = base_target + w_img * (base_target - ema_uncond)
                cls_target = base_target_cls + w_cls * (base_target_cls - ema_cls_uncond)

            # truth guidance (cond): base_target + w * (base_target - ema_cond)
            elif self.guidance_mode == "truth_cond":
                with torch.no_grad():
                    y_cond = labels

                    ema_input = model_input
                    ema_cls_input = cls_input
                    ema_t = time_input.flatten()
                    ema_y = y_cond

                    ema_cond, _, ema_cls_cond = self.ema_model(ema_input, ema_t, ema_y, cls_token=ema_cls_input)

                model_target = base_target + w_img * (base_target - ema_cond)
                cls_target = base_target_cls + w_cls * (base_target_cls - ema_cls_cond)

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

        # structural autocorrelation alignment, exclude CLS
        struc_loss = 0.0
        for z, z_tilde in zip(zs, zs_tilde):
            z_nc = _drop_cls(z)            # [B,N,D]
            h_nc = _drop_cls(z_tilde)
            A_enc = _autocorr(z_nc)
            A_den = _autocorr(h_nc)
            struc_loss += ((A_enc - A_den) ** 2).mean()
        struc_loss /= len(zs)
        
        return denoising_loss, proj_loss, time_input, noises, denoising_loss_cls, struc_loss
