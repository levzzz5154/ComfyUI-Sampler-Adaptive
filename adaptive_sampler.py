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
    base_step_size=0.0004,
    min_step_size=0.01,
    max_step_size=0.2,
    max_steps=100,
    smoothing_coef=0.0,
    sigma_schedule_out=None,
    denoise=1.0,
    error_bias=0.002,
):
    """Adaptive sampler that wraps another sampler with adaptive step sizes."""
    
    extra_args = {} if extra_args is None else extra_args

    disable_pbar = disable if disable is not None else not comfy.utils.PROGRESS_BAR_ENABLED

    model_sampling = model.inner_model.inner_model.model_sampling
    sigma_min = model_sampling.sigma_min
    sigma_max = model_sampling.sigma_max

    current_sigma = sigma_min + (sigma_max - sigma_min) * denoise
    step_size = min_step_size  # First step = smallest

    total_steps = 0
    error_val = 1.0
    v_prev = None
    
    # Track sigma schedule
    sigma_schedule = [current_sigma]

    denoised_capture = [None]

    def capture_denoised(inner_callback):
        def wrapper(info):
            denoised_capture[0] = info.get('denoised', None)
            if inner_callback:
                inner_callback(info)
        return wrapper

    while current_sigma > sigma_min and total_steps < max_steps:
        # Run wrapped sampler for one step
        sigma_target = max(current_sigma - step_size, sigma_min)
        
        # Create a single-step sigma tensor for the wrapped sampler
        step_sigmas = torch.tensor([current_sigma, sigma_target], device=x.device, dtype=x.dtype)
        
        x_prev = x.clone()
        
        # Run wrapped sampler (e.g., euler, heun)
        wrapped_callback = capture_denoised(callback)
        x = wrapped_sampler.sampler_function(model, x, step_sigmas, extra_args, wrapped_callback, disable_pbar)

        # Get velocity: v = denoised - x
        denoised = denoised_capture[0]
        if denoised is not None:
            v_current = denoised - x
        else:
            v_current = x - x_prev

        if callback is not None:
            callback({
                'x': x,
                'i': total_steps,
                'sigma': sigma_target,
                'sigma_hat': sigma_target,
                'denoised': denoised if denoised is not None else x
            })

        if not disable_pbar:
            if total_steps > 0:
                print(f"Step {total_steps + 1}: sigma={sigma_target:.6f}, step_size={step_size:.6f}, error={error_val:.6f}")
            else:
                print(f"Step {total_steps + 1}: sigma={sigma_target:.6f}, step_size={step_size:.6f}")

        # Compute error from velocity change
        if v_prev is not None:
            if error_type == "cosine":
                error = 1.0 - get_cosine_similarity(v_prev, v_current, dim=1).abs().mean()
            else:
                error = F.mse_loss(v_prev, v_current)

            error_val = max(error.item(), 1e-10)

            # Adjust step size for next iteration: base_step_size / error
            new_step_size = base_step_size / (error_val + error_bias)
            step_size = max(min_step_size, min(max_step_size, step_size * smoothing_coef + new_step_size * (1 - smoothing_coef)))

        v_prev = v_current

        current_sigma = sigma_target
        sigma_schedule.append(sigma_target)
        total_steps += 1

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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength (1.0 = full)"}),
                "error_type": (["cosine", "mse"], {"default": "cosine", "tooltip": "cosine: measure direction change, mse: measure magnitude change"}),
                "base_step_size": ("FLOAT", {"default": 0.0004, "min": 0.0001, "max": 1.0, "step": 0.0001, "tooltip": "Base multiplier for step size calculation"}),
                "min_step_size": ("FLOAT", {"default": 0.005, "min": 0.0001, "max": 1.0, "step": 0.0001, "tooltip": "Minimum allowed step size"}),
                "max_step_size": ("FLOAT", {"default": 0.2, "min": 0.001, "max": 1.0, "step": 0.001, "tooltip": "Maximum allowed step size"}),
                "max_steps": ("INT", {"default": 100, "min": 1, "max": 10000, "tooltip": "Maximum number of adaptive steps"}),
                "smoothing_coef": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "EMA coefficient: 0 = new value only, 1 = keep old value"}),
                "error_bias": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Bias added to error. Higher values = smaller steps"}),
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
        denoise=1.0,
        error_type="cosine",
        base_step_size=0.0004,
        min_step_size=0.01,
        max_step_size=0.2,
        max_steps=100,
        smoothing_coef=0.0,
        error_bias=0.002,
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
        sigma_min = model_sampling.sigma_min
        sigma_max = model_sampling.sigma_max
        start_sigma = sigma_min + (sigma_max - sigma_min) * denoise
        sigmas = torch.tensor([start_sigma, 0.0], device=x.device, dtype=x.dtype)

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
                smoothing_coef=smoothing_coef,
                sigma_schedule_out=sigma_schedule_out,
                denoise=denoise,
                error_bias=error_bias,
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
