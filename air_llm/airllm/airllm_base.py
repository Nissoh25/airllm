
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .profiler import LayeredProfiler

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False


# ---- transformers meta-device compatibility (2026-11-24) ----------------------------------
# AirLLM keeps the full model skeleton on the meta device between layer executions.
# transformers' mask builders (sdpa_mask / flex attention) probe mask CONTENT with scalar
# ops like `padding_mask.all()` to decide whether the causal fast-path applies. Scalar ops
# are illegal on meta tensors ("Tensor.item() cannot be called on meta tensors"), so such a
# probe crashes the forward. A meta mask can only be the default all-True padding mask built
# during meta-init (real user masks are always created on a concrete device), and an
# all-True mask is exactly the "no padding" case, which is also what `padding_mask=None`
# means to those probes. So we substitute None for meta masks inside the two probes.
try:
    import transformers.masking_utils as _masking_utils

    def _meta_safe(orig):
        def wrapper(padding_mask, *args, **kwargs):
            if padding_mask is not None and getattr(padding_mask, 'device', None) is not None \
                    and padding_mask.device.type == 'meta':
                padding_mask = None
            return orig(padding_mask, *args, **kwargs)
        return wrapper

    _masking_utils._ignore_causal_mask_sdpa = _meta_safe(_masking_utils._ignore_causal_mask_sdpa)
    _masking_utils._ignore_bidirectional_mask_sdpa = _meta_safe(_masking_utils._ignore_bidirectional_mask_sdpa)
except (ImportError, AttributeError):
    pass
# ----------------------------------------------------------------------------------------------


class AirLLMBaseModel:
    """
    Memory-frugal wrapper around a Hugging Face ``*ForCausalLM`` model.

    The checkpoint is split into per-layer shards on disk. The real transformers model is
    instantiated on the ``meta`` device (no memory used) and owns the full forward / generation
    logic. AirLLM only attaches forward hooks to each big module (embeddings, every decoder
    layer, the final norm and the lm_head) to stream that module's weights disk -> GPU right
    before it runs and free them right after, prefetching the next module on a worker thread.

    Because transformers drives the forward pass, AirLLM no longer needs to track per-architecture
    attention/rotary/cache details: new model architectures work as soon as transformers supports
    them.
    """

    # Subclasses override this to point at non-standard module names.
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                                 'layer_prefix': 'model.layers',
                                 'norm': 'model.norm',
                                 'lm_head': 'lm_head'}

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False):
        """
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
            modern models). float16 has too narrow a range for very deep models and overflows to
            inf/NaN, which silently corrupts the output, so we don't force it.
        max_seq_len : int, optional
            max seq length, by default 512
        layer_shards_saving_path : str, optional
            optional path to save the splitted shards, by default next to the model cache
        profiling_mode : bool, optional
            whether to profile the model loading time, default False
        compression: str, optional
            '4bit' or '8bit' to enable block-wise quantization of the on-disk shards
        hf_token: str, optional
            huggingface api token
        prefetching: bool, optional
            overlap the next layer's disk load with the current layer's compute
        delete_original: bool, optional
            delete the original downloaded checkpoint after splitting to save disk space
        """

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self.hf_quantizer = None

        if compression is not None and not bitsandbytes_installed:
            raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. '
                              'To use compression, please install bitsandbytes: `pip install bitsandbytes`')

        self.compression = compression
        self.hf_token = hf_token

        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original)

        self.running_device = device
        self.device = torch.device(self.running_device)

        # Prefer transformers' native implementation; only trust the model's bundled remote code when
        # transformers doesn't recognize the architecture. Vendored remote code is frequently pinned
        # to an old transformers and breaks against the current cache/generation APIs (e.g.
        # DeepSeek-V2's modeling_deepseek.py calls the long-removed DynamicCache.seen_tokens).
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=False, **token_kwargs)
            self.trust_remote_code = False
        except Exception:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=True, **token_kwargs)
            self.trust_remote_code = True

        # Default to the model's native dtype (bf16 for most modern models). Forcing fp16 overflows
        # on deep models (e.g. Qwen3-235B's 94 layers) and produces garbage; bf16's wider range
        # avoids it. Users can still override via dtype=.
        if dtype is None:
            cfg_dtype = getattr(self.config, "torch_dtype", None)
            if isinstance(cfg_dtype, str):
                cfg_dtype = getattr(torch, cfg_dtype, None)
            dtype = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.float16
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        # prefetch executor / state
        self.prefetching = prefetching
        if self.compression is not None and self.prefetching:
            print("prefetching is not supported together with compression for now; disabling prefetching.")
            self.prefetching = False
        self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetching else None
        self._prefetch_future = None
        self._prefetched_idx = None

        self.init_model()

        # compute layer count from the instantiated model
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count = len(model_attr)

        self.layer_names = [self.layer_names_dict['embed']] + \
                           [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.set_layers_from_layer_names()
        self._install_streaming_hooks()

    # ---- customization hooks for subclasses -------------------------------------------------

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    # ---- model construction -----------------------------------------------------------------

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        self.model = None
        try:
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="sdpa", trust_remote_code=self.trust_remote_code)
        except (ValueError, TypeError) as e:
            print(f"attn_implementation='sdpa' not available ({e}), falling back to eager attention")
            self.model = None
        if self.model is None:
            # Some (often remote-code) architectures don't support sdpa and also default to it, so we
            # must request eager explicitly; otherwise transformers re-selects sdpa and errors again.
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="eager", trust_remote_code=self.trust_remote_code)

        quantization_config = getattr(self.config, "quantization_config", None)
        # Pre-quantized FP8 (qwen3_5-style checkpoints): the finegrained-fp8 hub kernel hits
        # CUDA IMA on this build; the working path dequants each layer to the running dtype at
        # stream time (see move_layer_to_device). Skip the quantizer so weights go through the
        # plain placement branch. Other pre-quantized schemes (gptq/bnb) keep the quantizer path.
        if quantization_config is not None and \
                str(getattr(quantization_config, "quant_method", "")).lower() != "fp8":
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()
        self.model.generation_config = self.generation_config

        # Move all (already-materialized) buffers to the running device, preserving their dtype.
        # This includes rotary inv_freq, which transformers computes once at the model level and
        # passes down to every decoder layer.
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != 'meta':
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer)

        # Force the model to report the running (cuda) device even though its parameters live on
        # meta between layer executions, so transformers' generation utilities place inputs/cache
        # tensors on the right device.
        self._patch_device_property()

    def _patch_device_property(self):
        running_device = torch.device(self.running_device)
        running_dtype = self.running_dtype
        base_cls = type(self.model)

        class _AirLLMRuntimeModel(base_cls):
            @property
            def device(self):
                return running_device

            @property
            def dtype(self):
                return running_dtype

        self.model.__class__ = _AirLLMRuntimeModel

    def set_layers_from_layer_names(self):
        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    # ---- weight streaming -------------------------------------------------------------------

    def load_layer_to_cpu(self, layer_name):
        t = time.time()
        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)
            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        if self.prefetching and str(self.running_device).startswith("cuda") and torch.cuda.is_available():  # SPARK-PATCH UMA: pin_memory allocates CUDA ctx — CPU lane must skip
            for k in state_dict.keys():
                state_dict[k].pin_memory()

        return state_dict

    def move_layer_to_device(self, state_dict):
        moved = []
        for param_name in self._param_names_from_state_dict(state_dict):
            # fp8 block scales: only meaningful for in-place fp8 compute, which is broken on this
            # build (Triton IMA); weights are dequanted to the running dtype at stream time instead.
            if param_name.endswith("_scale_inv"):
                continue
            if self.hf_quantizer is not None and self._needs_quantization(param_name):
                # On-the-fly-quantizing schemes (e.g. bitsandbytes) reconstruct the param from the
                # weight plus companion quant-state tensors carried in state_dict.
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name,
                                                         self.running_device, state_dict)
            else:
                # FP8 pre-quantized weights: dequant to the running dtype at stream time.
                # The on-disk shard stays fp8 (memory compression); the forward pass runs in the
                # running dtype, which is what transformers' native model code expects.
                value = state_dict[param_name]
                if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    # SPARK-PATCH 2026-11-24: block-quantized FP8 (compressed-tensors layout)
                    # carries a companion "<name>_scale_inv" of shape [R//br, C//bc]. A plain
                    # f8->bf16 cast silently yields wrong values (raw codes, no scaling), so
                    # apply the block scale on CPU (source is pinned) before the device move.
                    scale = state_dict.get(param_name + '_scale_inv')
                    if scale is not None and scale.numel() > 1:
                        r, c = value.shape
                        sr, sc = scale.shape
                        # SPARK-PATCH 2026-11-24 (v2): dequant on the TARGET device when it is
                        # CUDA. The CPU path (value.float() -> f32 temp) cost ~20s/layer on GB10
                        # because 1.5GiB of f32 view+scale-broadcast runs at DRAM bandwidth on
                        # the CPU; on the Blackwell GPU the same op is ~100x faster and the
                        # 1.9GiB transient fits the unified pool. CPU fallback kept for
                        # non-CUDA running devices.
                        if str(self.running_device).startswith('cuda'):
                            dev = torch.device(self.running_device)
                            w = value.to(dev, non_blocking=True)
                            s = scale.to(dev, non_blocking=True)
                            w = w.float().view(sr, r // sr, sc, c // sc)
                            w = w * s.float().view(sr, 1, sc, 1)
                            value = w.view(r, c).to(self.running_device, self.running_dtype)
                            del w, s
                        else:
                            w = value.float().view(sr, r // sr, sc, c // sc)
                            w = w * scale.float().view(sr, 1, sc, 1)
                            value = w.view(r, c).to(self.running_dtype)
                    else:
                        value = value.to(self.running_dtype)
                set_module_tensor_to_device(self.model, param_name, self.running_device,
                                            value=value, dtype=self.running_dtype)
            moved.append(param_name)
        return moved

    def _needs_quantization(self, param_name):
        q = self.hf_quantizer
        # transformers renamed check_quantized_param -> param_needs_quantization.
        if hasattr(q, "param_needs_quantization"):
            return q.param_needs_quantization(self.model, param_name)
        return q.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})

    def _param_names_from_state_dict(self, state_dict):
        names = []
        for param_name in state_dict.keys():
            # bitsandbytes stores a weight plus companion quant-state tensors named
            # "<weight>.4bit.*" / "<weight>.8bit.*"; those are reconstructed together via
            # create_quantized_param, so collapse them down to the base weight name. Everything
            # else (including fp8 weight + weight_scale_inv pairs) is kept as distinct params.
            if '.4bit.' in param_name or '.8bit.' in param_name:
                base = param_name.split('.4bit.')[0].split('.8bit.')[0]
                if base not in names:
                    names.append(base)
            elif param_name not in names:
                names.append(param_name)
        return names

    def _install_streaming_hooks(self):
        # Modules execute in this order during a forward: embed -> layers -> norm -> lm_head.
        n = len(self.layer_names)

        # Detect tied input/output embeddings. When tied, lm_head shares the embedding weight, so
        # there is no separate lm_head shard. We keep the embedding resident on the GPU (it is the
        # only copy and such models are small) and re-tie lm_head to it, then stream only the
        # decoder layers and the final norm.
        self.tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", False))

        if self.tie_word_embeddings:
            embed_state = self.load_layer_to_cpu(self.layer_names[0])
            self.move_layer_to_device(embed_state)
            self.model.tie_weights()
            self._streamed_indices = list(range(1, n - 1))  # decoder layers + final norm
        else:
            self._streamed_indices = list(range(n))

        self._streamed_set = set(self._streamed_indices)

        for idx in self._streamed_indices:
            module = self.layers[idx]
            module._airllm_idx = idx
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    def _next_streamed_idx(self, idx):
        nxt = idx + 1
        return nxt if nxt in self._streamed_set else None

    def _pre_hook(self, module, args):
        idx = module._airllm_idx

        if self.prefetching and self._prefetch_future is not None and self._prefetched_idx == idx:
            state_dict = self._prefetch_future.result()
            self._prefetch_future = None
        else:
            state_dict = self.load_layer_to_cpu(self.layer_names[idx])

        module._airllm_moved = self.move_layer_to_device(state_dict)

        if self.prefetching:
            nxt = self._next_streamed_idx(idx)
            if nxt is not None:
                self._prefetch_future = self._executor.submit(self.load_layer_to_cpu, self.layer_names[nxt])
                self._prefetched_idx = nxt

    def _post_hook(self, module, args, output):
        if self.hf_quantizer is not None:
            for param_name in getattr(module, '_airllm_moved', []):
                set_module_tensor_to_device(self.model, param_name, 'meta')
        else:
            module.to('meta')
        clean_memory()
        return output

    # ---- delegation to the underlying transformers model ------------------------------------

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
