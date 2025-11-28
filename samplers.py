import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t):
    """vt: velocity prediction, xt: current x_t, t: [B]"""
    t = expand_t_like_x(t, xt)

    alpha_t = 1 - t
    sigma_t = t
    d_alpha_t = -torch.ones_like(xt, device=xt.device)
    d_sigma_t = torch.ones_like(xt, device=xt.device)

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score

def model_out_to_velocity(model_out, xt, t, prediction="v"):
    """
    model_out: raw output of the network (v or x depending on prediction)
    xt: current x_t
    t: [batch] time tensor (before expand)
    """
    if prediction == "v":
        return model_out

    elif prediction == "x":
        # v = (x_t - x_pred) / t  for your linear path
        t_expanded = expand_t_like_x(t, xt)
        t_safe = t_expanded.clamp(min=1e-5)
        return (xt - model_out) / t_safe

    else:
        raise NotImplementedError(f"Unknown prediction mode: {prediction}")


def compute_diffusion(t_cur):
    return 2 * t_cur


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        prediction="v",
        cls_latents=None,
        args=None
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        #[1000, 1000]
    _dtype = latents.dtype


    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    cls_x_next = cls_latents.to(torch.float64)
    device = x_next.device


    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            cls_x_cur = cls_x_next

            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                cls_model_input = torch.cat([cls_x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                cls_model_input = cls_x_cur
                y_cur = y

            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)

            eps_i = torch.randn_like(x_cur).to(device)
            cls_eps_i = torch.randn_like(cls_x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))
            cls_deps = cls_eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur, _, cls_v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs, cls_token=cls_model_input.to(dtype=_dtype)
                )
            # convert raw model outputs to velocities depending on prediction type
            v_cur = model_out_to_velocity(v_cur, model_input, time_input, prediction=prediction)
            cls_v_cur = model_out_to_velocity(cls_v_cur, cls_model_input, time_input, prediction=prediction)
            v_cur = v_cur.to(torch.float64)
            cls_v_cur = cls_v_cur.to(torch.float64)

            s_cur = get_score_from_velocity(v_cur, model_input, time_input)
            d_cur = v_cur - 0.5 * diffusion * s_cur

            cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input)
            cls_d_cur = cls_v_cur - 0.5 * diffusion * cls_s_cur

            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

                cls_d_cur_cond, cls_d_cur_uncond = cls_d_cur.chunk(2)
                if args.cls_cfg_scale >0:
                    cls_d_cur = cls_d_cur_uncond + args.cls_cfg_scale * (cls_d_cur_cond - cls_d_cur_uncond)
                else:
                    cls_d_cur = cls_d_cur_cond
            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
            cls_x_next = cls_x_cur + cls_d_cur * dt + torch.sqrt(diffusion) * cls_deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    cls_x_cur = cls_x_next

    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        cls_model_input = torch.cat([cls_x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        cls_model_input = cls_x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    v_cur, _, cls_v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs, cls_token=cls_model_input.to(dtype=_dtype)
        )
    # convert raw model outputs to velocities depending on prediction type
    v_cur = model_out_to_velocity(v_cur, model_input, time_input, prediction=prediction)
    cls_v_cur = model_out_to_velocity(cls_v_cur, cls_model_input, time_input, prediction=prediction)
    v_cur = v_cur.to(torch.float64)
    cls_v_cur = cls_v_cur.to(torch.float64)


    s_cur = get_score_from_velocity(v_cur, model_input, time_input)
    cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input)

    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    cls_d_cur = cls_v_cur - 0.5 * diffusion * cls_s_cur  # d_cur [b, 4, 32 ,32]

    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        cls_d_cur_cond, cls_d_cur_uncond = cls_d_cur.chunk(2)
        if args.cls_cfg_scale > 0:
            cls_d_cur = cls_d_cur_uncond + args.cls_cfg_scale * (cls_d_cur_cond - cls_d_cur_uncond)
        else:
            cls_d_cur = cls_d_cur_cond

    mean_x = x_cur + dt * d_cur
    cls_mean_x = cls_x_cur + dt * cls_d_cur

    return mean_x
