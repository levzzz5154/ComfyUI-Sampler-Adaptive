from .adaptive_sampler import AdaptiveSamplerCustom, sample_adaptive_custom


NODE_CLASS_MAPPINGS = {
    "AdaptiveSamplerCustom": AdaptiveSamplerCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveSamplerCustom": "Adaptive Sampler Custom",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
