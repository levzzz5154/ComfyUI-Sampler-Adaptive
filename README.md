# ComfyUI-Sampler-Adaptive

Adaptive sampling custom node for ComfyUI.

![Showcase](showcase.png)

See `anima_00321_.png` for an example workflow.

## Known Limitations

- Ancestral samplers (e.g., `euler_ancestral`, `dpmpp_2s_ancestral`) do not work correctly at this time.
- Samplers from RES4LYF are not supported.
- Multistep/stateful samplers (e.g., `dpmpp_2m`, `lms`, `ipndm`, `dpmpp_2m_sde`) are not supported. They will produce output equivalent to first-order samplers. Use `euler`, `heun`, `dpm_2`, or other stateless samplers instead.

## Parameters

- **error_type**: Cosine or MSE similarity for error calculation between steps
- **base_step_size**: Base multiplier for step size adaptation
- **min_step_size**: Minimum allowed step size
- **max_step_size**: Maximum allowed step size
- **max_steps**: Maximum number of adaptive steps
- **smoothing_coef**: EMA coefficient for step size updates. 0.0 = use only new calculated step size, 1.0 = keep previous step size unchanged, 0.5 = equal blend of old and new
- **error_bias**: Bias added to error value in step size calculation (default: 0.0). Higher values result in smaller step sizes.
