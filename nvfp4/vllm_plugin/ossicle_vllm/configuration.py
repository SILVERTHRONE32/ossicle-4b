from transformers import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class OssicleConfig(PretrainedConfig):
    model_type = "ossicle"

    def __init__(self, audio_cfg=None, llm_cfg=None, has_stereo=False, stereo_in_dim=72, stereo_hidden=512, initializer_range=0.02, reader_cfg=None, quantization=None, **kwargs):
        self.audio_cfg = audio_cfg or {}
        self.llm_cfg = llm_cfg or {}
        self.has_stereo, self.stereo_in_dim, self.stereo_hidden = has_stereo, stereo_in_dim, stereo_hidden
        self.initializer_range, self.reader_cfg, self.quantization = initializer_range, reader_cfg, quantization
        self.text_config = Qwen3_5TextConfig(**self.llm_cfg) if self.llm_cfg else None     # vLLM reads the language-model settings from here
        super().__init__(**kwargs)
