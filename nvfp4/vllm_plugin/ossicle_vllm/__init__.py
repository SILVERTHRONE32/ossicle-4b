def register():
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .configuration import OssicleConfig
    AutoConfig.register("ossicle", OssicleConfig, exist_ok=True)
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig
    MODELS_CONFIG_MAP["OssicleForConditionalGeneration"] = Qwen3_5ForConditionalGenerationConfig
    ModelRegistry.register_model("OssicleForConditionalGeneration", "ossicle_vllm.model:OssicleForConditionalGeneration")
