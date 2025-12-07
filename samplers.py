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

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


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
        path_type="linear",
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
            v_cur = v_cur.to(torch.float64)
            cls_v_cur = cls_v_cur.to(torch.float64)

            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur

            cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input, path_type=path_type)
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
    v_cur = v_cur.to(torch.float64)
    cls_v_cur = cls_v_cur.to(torch.float64)


    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input, path_type=path_type)

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

def euler_maruyama_sampler_path_drop(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
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

            use_cfg = cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low

            if use_cfg:
                time_input = torch.ones(x_cur.size(0)).to(device=device, dtype=torch.float64) * t_cur
                # Conditional branch (no path drop)
                v_cond, _, cls_v_cond = model(
                    x_cur.to(dtype=_dtype),
                    time_input.to(dtype=_dtype),
                    y=y,
                    cls_token=cls_x_cur.to(dtype=_dtype),
                    uncond=False,
                )
                # Unconditional branch (enable path drop)
                v_uncond, _, cls_v_uncond = model(
                    x_cur.to(dtype=_dtype),
                    time_input.to(dtype=_dtype),
                    y=y_null,
                    cls_token=cls_x_cur.to(dtype=_dtype),
                    uncond=True,
                )
            else:
                model_input = x_cur
                cls_model_input = cls_x_cur
                y_cur = y
                kwargs = dict(y=y_cur)
                time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur

                v_cur, _, cls_v_cur = model(
                    model_input.to(dtype=_dtype),
                    time_input.to(dtype=_dtype),
                    **kwargs,
                    cls_token=cls_model_input.to(dtype=_dtype),
                )
            diffusion = compute_diffusion(t_cur)

            eps_i = torch.randn_like(x_cur).to(device)
            cls_eps_i = torch.randn_like(cls_x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))
            cls_deps = cls_eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            if use_cfg:
                v_cond = v_cond.to(torch.float64)
                v_uncond = v_uncond.to(torch.float64)
                cls_v_cond = cls_v_cond.to(torch.float64)
                cls_v_uncond = cls_v_uncond.to(torch.float64)

                s_cond = get_score_from_velocity(v_cond, x_cur, time_input, path_type=path_type)
                d_cond = v_cond - 0.5 * diffusion * s_cond

                s_uncond = get_score_from_velocity(v_uncond, x_cur, time_input, path_type=path_type)
                d_uncond = v_uncond - 0.5 * diffusion * s_uncond

                d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)

                cls_s_cond = get_score_from_velocity(cls_v_cond, cls_x_cur, time_input, path_type=path_type)
                cls_d_cond = cls_v_cond - 0.5 * diffusion * cls_s_cond

                cls_s_uncond = get_score_from_velocity(cls_v_uncond, cls_x_cur, time_input, path_type=path_type)
                cls_d_uncond = cls_v_uncond - 0.5 * diffusion * cls_s_uncond

                if args.cls_cfg_scale > 0:
                    cls_d_cur = cls_d_uncond + args.cls_cfg_scale * (cls_d_cond - cls_d_uncond)
                else:
                    cls_d_cur = cls_d_cond
            else:
                v_cur = v_cur.to(torch.float64)
                cls_v_cur = cls_v_cur.to(torch.float64)

                s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
                d_cur = v_cur - 0.5 * diffusion * s_cur

                cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input, path_type=path_type)
                cls_d_cur = cls_v_cur - 0.5 * diffusion * cls_s_cur
            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
            cls_x_next = cls_x_cur + cls_d_cur * dt + torch.sqrt(diffusion) * cls_deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    cls_x_cur = cls_x_next

    use_cfg = cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low

    if use_cfg:
        time_input = torch.ones(x_cur.size(0)).to(device=device, dtype=torch.float64) * t_cur

        # Conditional branch (no path drop)
        v_cond, _, cls_v_cond = model(
            x_cur.to(dtype=_dtype),
            time_input.to(dtype=_dtype),
            y=y,
            cls_token=cls_x_cur.to(dtype=_dtype),
            uncond=False,
        )

        # Unconditional branch (enable path drop)
        v_uncond, _, cls_v_uncond = model(
            x_cur.to(dtype=_dtype),
            time_input.to(dtype=_dtype),
            y=y_null,
            cls_token=cls_x_cur.to(dtype=_dtype),
            uncond=True,
        )

        v_cond = v_cond.to(torch.float64)
        v_uncond = v_uncond.to(torch.float64)
        cls_v_cond = cls_v_cond.to(torch.float64)
        cls_v_uncond = cls_v_uncond.to(torch.float64)

        s_cond = get_score_from_velocity(v_cond, x_cur, time_input, path_type=path_type)
        s_uncond = get_score_from_velocity(v_uncond, x_cur, time_input, path_type=path_type)

        diffusion = compute_diffusion(t_cur)
        d_cond = v_cond - 0.5 * diffusion * s_cond
        d_uncond = v_uncond - 0.5 * diffusion * s_uncond

        d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)

        cls_s_cond = get_score_from_velocity(cls_v_cond, cls_x_cur, time_input, path_type=path_type)
        cls_s_uncond = get_score_from_velocity(cls_v_uncond, cls_x_cur, time_input, path_type=path_type)

        cls_d_cond = cls_v_cond - 0.5 * diffusion * cls_s_cond
        cls_d_uncond = cls_v_uncond - 0.5 * diffusion * cls_s_uncond

        if args.cls_cfg_scale > 0:
            cls_d_cur = cls_d_uncond + args.cls_cfg_scale * (cls_d_cond - cls_d_uncond)
        else:
            cls_d_cur = cls_d_cond
    else:
        model_input = x_cur
        cls_model_input = cls_x_cur
        y_cur = y
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(
            device=device, dtype=torch.float64
        ) * t_cur

        v_cur, _, cls_v_cur = model(
            model_input.to(dtype=_dtype),
            time_input.to(dtype=_dtype),
            **kwargs,
            cls_token=cls_model_input.to(dtype=_dtype),
        )
        v_cur = v_cur.to(torch.float64)
        cls_v_cur = cls_v_cur.to(torch.float64)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input, path_type=path_type)

        diffusion = compute_diffusion(t_cur)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        cls_d_cur = cls_v_cur - 0.5 * diffusion * cls_s_cur  # d_cur [b, 4, 32 ,32]

    mean_x = x_cur + dt * d_cur
    cls_mean_x = cls_x_cur + dt * cls_d_cur

    return mean_x