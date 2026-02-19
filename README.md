# ComfyUI-Sampler-Adaptive

Adaptive sigma scheduling custom node for ComfyUI. Dynamically adjusts step sizes during sampling based on how the denoising direction changes.

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

## How It Works

The adaptive sampler dynamically adjusts step sizes during sampling based on how much the "velocity" (direction of denoising) changes between steps:

1. **Velocity Calculation**: At each step, velocity is computed as `v = denoised - x` (the direction the latent is moving)

2. **Error Measurement**: The error is calculated by comparing the current velocity to the previous one:
   - **Cosine**: `1 - |cosine_similarity(v_prev, v_current)|` — measures directional change
   - **MSE**: Mean squared error between velocities — measures magnitude change

3. **Step Size Adaptation**: 
   - High error (velocity changing rapidly) → smaller steps for more precision
   - Low error (velocity stable) → larger steps for faster progress
   - Formula: `new_step_size = base_step_size / (error + error_bias)`

4. **EMA Smoothing**: The new step size is blended with the previous one to avoid sudden jumps:
   `step_size = smoothing_coef * old + (1 - smoothing_coef) * new`

This allows the sampler to take fewer steps in "easy" regions and more steps where the denoising direction is changing rapidly.

## Notes

This node has been mainly tested with the Anima model. Results may vary with other models.
