"""Ossicle in vLLM: the Qwen3-Omni audio encoder (with Ossicle's trained head as its output layers) feeds audio embeddings into a Qwen3.5 language model.
Prompt layout (as trained): <|im_start|>user\n[audio]{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n  with [audio] = one <|vision_pad|> placeholder."""
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc, MambaStateCopyFuncCalculator, MambaStateDtypeCalculator, MambaStateShapeCalculator
from vllm.model_executor.models.interfaces import IsHybrid, MultiModalEmbeddings, SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder, _get_feat_extract_output_lengths
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import BaseDummyInputsBuilder, BaseMultiModalProcessor, BaseProcessingInfo, PromptReplacement, PromptUpdate
from vllm.sequence import IntermediateTensors

from .stereo import cue_tokens

PLACEHOLDER = "<|vision_pad|>"
MAX_AUDIO_SECONDS = 600


class OssicleProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        return WhisperFeatureExtractor.from_pretrained(self.ctx.model_config.model)

    def get_data_parser(self):
        return MultiModalDataParser(target_sr=16000, target_channels=None, expected_hidden_size=self._get_expected_hidden_size(),
                                    allow_missing_mm_embeddings=self.allow_missing_mm_embeddings)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int] | None = None) -> Mapping[str, int]:
        if (mm_counts or {}).get("audio", 0) <= 0:
            return {}
        frames = torch.tensor([MAX_AUDIO_SECONDS * 100])
        return {"audio": int(_get_feat_extract_output_lengths(frames).item())}


class OssicleDummyInputsBuilder(BaseDummyInputsBuilder[OssicleProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return PLACEHOLDER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: MultiModalDummyOptions) -> MultiModalDataDict:
        return {"audio": self._get_dummy_audios(length=60 * 16000, num_audios=mm_counts.get("audio", 0), overrides=mm_options.get("audio"))}


class OssicleMultiModalProcessor(BaseMultiModalProcessor[OssicleProcessingInfo]):
    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str | None:
        return None

    def _call_hf_processor(self, hf_data: Mapping[str, object], hf_kwargs: Mapping[str, object]) -> BatchFeature:
        audios = hf_data.get("audio") or hf_data.get("audios") or []
        if not audios:
            return BatchFeature({}, tensor_type="pt")
        fe = self.info.get_feature_extractor()
        feats, lens, cues, ntoks = [], [], [], []
        for a in audios:
            a = np.asarray(a[0] if isinstance(a, tuple) else a, dtype=np.float32)
            if a.ndim == 2:
                ch = a.T if a.shape[0] > a.shape[1] else a            # (channels, samples)
            else:
                ch = a[None]
            mono = ch.mean(0)
            out = fe(mono, sampling_rate=16000, return_attention_mask=True, padding="longest", truncation=False, return_tensors="pt")
            n = int(out["attention_mask"].sum())
            feats.append(out["input_features"][0, :, :n]); lens.append(n)
            ntok = int(_get_feat_extract_output_lengths(torch.tensor([n])).item()); ntoks.append(ntok)
            rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
            stereo = ch.shape[0] >= 2 and rms(ch[0] - ch[1]) > 1e-3 * (rms(mono) + 1e-8)      # two genuinely different channels
            cues.append(torch.from_numpy(cue_tokens(ch[:2], ntok)) if stereo else torch.zeros(ntok, 72))
        return BatchFeature({"input_features": torch.cat(feats, dim=1), "audio_feature_lengths": torch.tensor(lens),
                             "stereo_cues": torch.cat(cues, dim=0), "audio_token_counts": torch.tensor(ntoks)}, tensor_type="pt")

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
        lens = hf_inputs.get("audio_feature_lengths", torch.empty((0,))).flatten()
        toks = hf_inputs.get("audio_token_counts", torch.empty((0,))).flatten()
        return dict(input_features=MultiModalFieldConfig.flat_from_sizes("audio", lens, dim=1),
                    audio_feature_lengths=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
                    stereo_cues=MultiModalFieldConfig.flat_from_sizes("audio", toks, dim=0),
                    audio_token_counts=MultiModalFieldConfig.batched("audio", keep_on_cpu=True))

    def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs: MultiModalKwargsItems) -> Sequence[PromptUpdate]:
        tok = self.info.get_tokenizer()
        pid = tok.convert_tokens_to_ids(PLACEHOLDER)
        lens = out_mm_kwargs.get_data().get("audio_feature_lengths")
        n_tokens = _get_feat_extract_output_lengths(lens.flatten()).tolist() if lens is not None else []
        return [PromptReplacement(modality="audio", target=[pid], replacement=lambda i: [pid] * int(n_tokens[i]))]


class StereoAdapter(nn.Module):
    """Zero-initialised MLP from interaural cues to the embedding size; the difference form makes identical channels contribute exactly zero."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden, out_dim, dtype=torch.float32)

    def forward(self, f):
        z = torch.zeros_like(f[:1])
        return self.fc2(torch.nn.functional.gelu(self.fc1(f))) - self.fc2(torch.nn.functional.gelu(self.fc1(z)))


@MULTIMODAL_REGISTRY.register_processor(OssicleMultiModalProcessor, info=OssicleProcessingInfo, dummy_inputs=OssicleDummyInputsBuilder)
class OssicleForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE, IsHybrid):
    packed_modules_mapping = {**Qwen3_5ForCausalLM.packed_modules_mapping, "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"]}

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype,
                                                                     vllm_config.cache_config.mamba_ssm_cache_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        hf = vllm_config.model_config.hf_text_config
        spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        return MambaStateShapeCalculator.gated_delta_net_state_shape(vllm_config.parallel_config.tensor_parallel_size, hf.linear_num_key_heads, hf.linear_num_value_heads,
                                                                     hf.linear_key_head_dim, hf.linear_value_head_dim, hf.linear_conv_kernel_dim, spec)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return PLACEHOLDER
        raise ValueError("Only audio is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        text_cfg = Qwen3_5TextConfig(**config.llm_cfg)
        audio_cfg = Qwen3OmniMoeAudioEncoderConfig(**{**config.audio_cfg, "output_dim": text_cfg.hidden_size})
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(audio_cfg, prefix=maybe_prefix(prefix, "audio_tower"))
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(vllm_config=vllm_config, hf_config=text_cfg, prefix=maybe_prefix(prefix, "language_model"),
                                                              architectures=["Qwen3_5ForCausalLM"])
        self.stereo_adapter = StereoAdapter(config.stereo_in_dim, config.stereo_hidden, text_cfg.hidden_size) if config.has_stereo else None
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Ossicle was trained with ordinary sequential positions (audio tokens included), so all rotary dimensions carry the same positions."""
        n = len(input_tokens)
        dims = len(self.config.text_config.rope_parameters.get("mrope_section", [0, 0, 0]))
        return torch.arange(n).unsqueeze(0).expand(dims, n).clone(), 0

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        feats, lens = kwargs.get("input_features"), kwargs.get("audio_feature_lengths")
        if feats is None:
            return []
        lens = lens.flatten().to(feats.device)
        out_lens = _get_feat_extract_output_lengths(lens)
        emb = self.audio_tower(feats.to(self.audio_tower.dtype), feature_lens=lens, aftercnn_lens=out_lens)
        cues = kwargs.get("stereo_cues")
        if self.stereo_adapter is not None and cues is not None and cues.shape[0] == emb.shape[0]:
            emb = emb + self.stereo_adapter(cues.to(emb.device, torch.float32)).to(emb.dtype)        # exactly zero for mono clips
        return emb.split(out_lens.tolist())

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor, intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None, **kwargs: object):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
