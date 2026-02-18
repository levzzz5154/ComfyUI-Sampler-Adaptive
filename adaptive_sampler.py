import torch
import torch.nn.functional as F

import comfy
import comfy.sample
import comfy.samplers
from comfy.samplers import KSAMPLER

import latent_preview


def get_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(start_dim=dim), b.flatten(start_dim=dim), dim=dim)


def sample_adaptive_custom(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    wrapped_sampler=None,
    error_type="cosine",
    base_step_size=1.0,
    min_step_size=0.01,
    max_step_size=1.0,
    max_steps=100,
    sigma_schedule_out=None,
):
    """Adaptive sampler that wraps another sampler with adaptive step sizes."""
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    disable_pbar = disable if disable is not None else not comfy.utils.PROGRESS_BAR_ENABLED

    model_sampling = model.inner_model.inner_model.model_sampling
    sigma_min = model_sampling.sigma_min
    sigma_max = model_sampling.sigma_max

    current_sigma = sigma_max
    step_size = min_step_size  # First step = smallest

    total_steps = 0
    eps_prev = None
    
    # Track sigma schedule
    sigma_schedule = [sigma_max]

    while current_sigma > sigma_min and total_steps < max_steps:
        if callback is not None:
            denoised = x + current_sigma * model(x, current_sigma * s_in, **extra_args)
            callback({
                'x': x,
                'i': total_steps,
                'sigma': current_sigma,
                'sigma_hat': current_sigma,
                'denoised': denoised
            })

        if not disable_pbar:
            if eps_prev is not None:
                print(f"Step {total_steps + 1}: sigma={current_sigma:.6f}, step_size={step_size:.6f}, error={error_val:.6f}")
            else:
                print(f"Step {total_steps + 1}: sigma={current_sigma:.6f}, step_size={step_size:.6f}")

        # Get current prediction
        eps_current = model(x, current_sigma * s_in, **extra_args)

        # Run wrapped sampler for one step
        sigma_target = max(current_sigma - step_size, sigma_min)
        dt = current_sigma - sigma_target

        # Create a single-step sigma tensor for the wrapped sampler
        step_sigmas = torch.tensor([current_sigma, sigma_target], device=x.device, dtype=x.dtype)
        
        # Run wrapped sampler (e.g., euler, heun)
        if wrapped_sampler is not None:
            # Call the sampler_function directly with proper args
            x = wrapped_sampler.sampler_function(model, x, step_sigmas, extra_args, None, True)
        else:
            # Fallback to simple Euler
            x = x - eps_current * dt

        # Get new prediction for error calculation
        eps_new = model(x, sigma_target * s_in, **extra_args)

        # Compute error between consecutive predictions
        if error_type == "cosine":
            error = 1.0 - get_cosine_similarity(eps_current, eps_new, dim=1).abs().mean()
        else:
            error = F.mse_loss(eps_current, eps_new)

        error_val = max(error.item(), 1e-10)

        # Adjust step size for next iteration: base_step_size / error
        step_size = base_step_size / error_val
        step_size = max(min_step_size, min(max_step_size, step_size))

        current_sigma = sigma_target
        sigma_schedule.append(sigma_target)
        total_steps += 1
        eps_prev = eps_new

    if not disable_pbar:
        print(f"Completed in {total_steps} steps, final sigma: {current_sigma:.6f}")

    if sigma_schedule_out is not None:
        sigma_schedule_out[0] = torch.tensor(sigma_schedule, device=x.device, dtype=x.dtype)

    return x


class AdaptiveSamplerCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "latent_image": ("LATENT",),
                "error_type": (["cosine", "mse"], {"default": "cosine"}),
                "base_step_size": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.001}),
                "min_step_size": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 10.0, "step": 0.0001}),
                "max_step_size": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.001}),
                "max_steps": ("INT", {"default": 100, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "SIGMAS")
    RETURN_NAMES = ("output", "denoised", "sigmas")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(
        self,
        noise,
        guider,
        sampler,
        latent_image,
        error_type="cosine",
        base_step_size=1.0,
        min_step_size=0.01,
        max_step_size=1.0,
        max_steps=100,
    ):
        latent = latent_image.copy()
        x = latent["samples"]
        
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x = comfy.sample.fix_empty_latent_channels(guider.model_patcher, x, latent.get("downscale_ratio_spacial", None))
        latent["samples"] = x

        # Create dummy sigmas for the wrapped sampler - we'll ignore them
        model_sampling = guider.model_patcher.get_model_object("model_sampling")
        sigma_max = model_sampling.sigma_max
        sigmas = torch.tensor([sigma_max, 0.0], device=x.device, dtype=x.dtype)

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, max_steps, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Store sigma schedule
        sigma_schedule_out = [None]

        def adaptive_sampler_function(model, x, sigmas, extra_args, callback, disable):
            return sample_adaptive_custom(
                model=model,
                x=x,
                sigmas=sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                wrapped_sampler=sampler,
                error_type=error_type,
                base_step_size=base_step_size,
                min_step_size=min_step_size,
                max_step_size=max_step_size,
                max_steps=max_steps,
                sigma_schedule_out=sigma_schedule_out,
            )

        sampler_obj = KSAMPLER(adaptive_sampler_function, {})

        noise_gen = noise.generate_noise(latent)
        samples = guider.sample(noise_gen, x, sampler_obj, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples

        if "x0" in x0_output:
            x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [s.shape for s in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        return (out, out_denoised, sigma_schedule_out[0])
