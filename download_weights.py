"""Download the tested LongCat-Video Avatar 1.5 model subset.

Downloads are resumable and idempotent through Hugging Face's snapshot cache.
Only files required by the validated single-audio avatar workflow are selected.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


AVATAR_REPOSITORY = "meituan-longcat/LongCat-Video-Avatar-1.5"
BASE_REPOSITORY = "meituan-longcat/LongCat-Video"

COMMON_IGNORE_PATTERNS = [
    "*.bin",
    "*.msgpack",
    "*.fp32.*",
    "pytorch_model.bin",
    "**/.cache/**",
]

AVATAR_ALLOW_PATTERNS = [
    "base_model_int8/config.json",
    "base_model_int8/quantization_config.json",
    "base_model_int8/quantized_model-*.safetensors",
    "base_model_int8/quantized_model.safetensors.index.json",
    "lora/dmd_lora.safetensors",
    "scheduler/scheduler_config.json",
    "vocal_separator/Kim_Vocal_2.onnx",
    "vocal_separator/download_checks.json",
    "vocal_separator/mdx_model_data.json",
    "vocal_separator/vr_model_data.json",
    "whisper-large-v3/config.json",
    "whisper-large-v3/generation_config.json",
    "whisper-large-v3/merges.txt",
    "whisper-large-v3/model.safetensors",
    "whisper-large-v3/normalizer.json",
    "whisper-large-v3/preprocessor_config.json",
    "whisper-large-v3/special_tokens_map.json",
    "whisper-large-v3/tokenizer.json",
    "whisper-large-v3/tokenizer_config.json",
    "whisper-large-v3/vocab.json",
]

BASE_ALLOW_PATTERNS = [
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-*.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def download_repository(repository, destination, allow_patterns):
    target = destination / repository.rsplit("/", 1)[-1]
    print(f"Downloading {repository} to {target}", flush=True)
    snapshot_download(
        repo_id=repository,
        repo_type="model",
        local_dir=str(target),
        allow_patterns=allow_patterns,
        ignore_patterns=COMMON_IGNORE_PATTERNS,
    )
    print(f"Finished {repository}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/workspace/weights"),
        help="Persistent directory for downloaded models",
    )
    parser.add_argument("--avatar-only", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    options = parser.parse_args()
    if options.avatar_only and options.base_only:
        parser.error("--avatar-only and --base-only cannot be combined")
    return options


def main():
    options = parse_args()
    options.destination.expanduser().mkdir(parents=True, exist_ok=True)

    if not options.base_only:
        download_repository(
            AVATAR_REPOSITORY, options.destination, AVATAR_ALLOW_PATTERNS
        )
    if not options.avatar_only:
        download_repository(
            BASE_REPOSITORY, options.destination, BASE_ALLOW_PATTERNS
        )


if __name__ == "__main__":
    main()