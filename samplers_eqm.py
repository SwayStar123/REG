import torch


@torch.no_grad()
def eqm_gradient_descent_sampler(
        model,
        latents,
        y,
        num_steps=250,
        step_size=0.003,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        cls_latents=None,
        args=None,
        use_nag=False,
        nag_momentum=0.35,
):
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)

    _dtype = latents.dtype
    x = latents.to(torch.float64)
    cls_x = cls_latents.to(torch.float64) if cls_latents is not None else None
    device = x.device

    x_prev = x.clone() if use_nag else None
    cls_x_prev = cls_x.clone() if (use_nag and cls_x is not None) else None

    def get_gradient(x_cur, cls_x_cur, y_cur):
        dummy_time = torch.zeros(x_cur.size(0), device=device, dtype=_dtype)
        grad, _, cls_grad = model(
            x_cur.to(dtype=_dtype),
            dummy_time,
            y=y_cur,
            cls_token=cls_x_cur.to(dtype=_dtype) if cls_x_cur is not None else None,
        )
        grad = grad.to(torch.float64)
        cls_grad = cls_grad.to(torch.float64) if cls_grad is not None else None
        return grad, cls_grad

    for step in range(num_steps):
        if use_nag and step > 0:
            x_eval = x + nag_momentum * (x - x_prev)
            cls_x_eval = cls_x + nag_momentum * (cls_x - cls_x_prev) if cls_x is not None else None
        else:
            x_eval = x
            cls_x_eval = cls_x

        if cfg_scale > 1.0:
            x_input = torch.cat([x_eval, x_eval], dim=0)
            cls_x_input = torch.cat([cls_x_eval, cls_x_eval], dim=0) if cls_x_eval is not None else None
            y_input = torch.cat([y, y_null], dim=0)

            grad, cls_grad = get_gradient(x_input, cls_x_input, y_input)

            grad_cond, grad_uncond = grad.chunk(2)
            grad = grad_uncond + cfg_scale * (grad_cond - grad_uncond)

            if cls_grad is not None:
                cls_grad_cond, cls_grad_uncond = cls_grad.chunk(2)
                if args is not None and getattr(args, "cls_cfg_scale", 0) > 0:
                    cls_grad = cls_grad_uncond + args.cls_cfg_scale * (cls_grad_cond - cls_grad_uncond)
                else:
                    cls_grad = cls_grad_cond
        else:
            grad, cls_grad = get_gradient(x_eval, cls_x_eval, y)

        if use_nag:
            x_prev = x.clone()
            if cls_x is not None:
                cls_x_prev = cls_x.clone()

        x = x - step_size * grad
        if cls_x is not None and cls_grad is not None:
            cls_x = cls_x - step_size * cls_grad

    return x.to(_dtype)
