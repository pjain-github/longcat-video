import os
import gc
import json
import shutil
from typing import Optional, Set

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


def print_ram(msg):
    rss = psutil.Process().memory_info().rss / 1024**3
    print(f"[RAM] {msg}: {rss:.2f} GB")


class QuantizedLinear(nn.Module):
    """INT8 weight-only quantized linear layer with per-channel symmetric quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = x.dtype
        weight = self.weight_int8.to(compute_dtype) * self.weight_scale.to(compute_dtype).unsqueeze(1)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, bias=has_bias)

        weight = linear.weight.data.float()
        scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        ql.weight_int8 = weight_int8
        ql.weight_scale = scale

        if has_bias:
            ql.bias = linear.bias.data.to(torch.bfloat16)

        return ql


DEFAULT_SKIP_PATTERNS = {
    "final_layer.linear",
}


def load_quantized_dit(checkpoint_dir: str, subfolder: str = "base_model_int8", **kwargs):
    """Load a quantized DiT model one safetensors shard at a time."""
    from .avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    quantized_dir = os.path.join(checkpoint_dir, subfolder)
    config_path = os.path.join(quantized_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    for key in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(key, None)
    config.update(kwargs)

    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)

    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(
            pattern in name for pattern in DEFAULT_SKIP_PATTERNS
        ):
            modules_to_replace[name] = QuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )

    for name, quantized in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], quantized)

    expected = dict(model.named_parameters())
    expected.update(model.named_buffers())

    def assign_tensor(key, tensor):
        module_name, tensor_name = key.rsplit(".", 1) if "." in key else ("", key)
        module = model if not module_name else model.get_submodule(module_name)
        if tensor_name in module._parameters:
            parameter = module._parameters[tensor_name]
            module._parameters[tensor_name] = nn.Parameter(
                tensor, requires_grad=parameter.requires_grad
            )
        elif tensor_name in module._buffers:
            module._buffers[tensor_name] = tensor
        else:
            raise KeyError(f"Checkpoint tensor {key!r} is not a model parameter or buffer")

    index_path = os.path.join(quantized_dir, "quantized_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(
            file_name
            for file_name in os.listdir(quantized_dir)
            if file_name.endswith(".safetensors") and "index" not in file_name
        )

    loaded_keys = set()
    for shard_file in shard_files:
        shard_dict = load_file(os.path.join(quantized_dir, shard_file), device="cpu")
        unknown_keys = set(shard_dict) - set(expected)
        if unknown_keys:
            raise RuntimeError(
                f"Unexpected checkpoint tensors: {sorted(unknown_keys)[:8]}"
            )
        for key, tensor in shard_dict.items():
            assign_tensor(key, tensor)
            loaded_keys.add(key)
        del shard_dict

    missing_keys = set(expected) - loaded_keys
    if missing_keys:
        raise RuntimeError(f"Missing checkpoint tensors: {sorted(missing_keys)[:8]}")

    model.eval()
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            continue
        for _, parameter in module.named_parameters(recurse=False):
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)

    return model
