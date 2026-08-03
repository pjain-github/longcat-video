"""Repeatable dependency installer for LongCat-Video.

This file is intentionally standalone so it can bootstrap a fresh server
without requiring setuptools or LongCat's requirements files first.

Example:
    python setup.py install_requirements --avatar
"""

import argparse
from pathlib import Path
import subprocess
import sys


PYTORCH_INDEXES = {
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}
PYPI_INDEX = "https://pypi.org/simple"

TORCH_RUNTIME_REQUIREMENTS = [
    "filelock",
    "fsspec",
    "jinja2",
    "nvidia-cusparselt-cu12==0.6.2",
    "networkx",
    "opt-einsum",
    "pillow",
    "sympy==1.13.1",
    "triton==3.2.0",
    "typing-extensions>=4.10.0",
]

BASE_REQUIREMENTS = [
    "huggingface_hub==0.36.2",
    "numpy==1.26.4",
    "transformers==4.41.0",
    "loguru==0.7.2",
    "diffusers==0.35.1",
    "einops==0.8.0",
    "ftfy==6.2.0",
    "psutil==6.0.0",
    "av==12.0.0",
    "opencv-python==4.9.0.80",
    "streamlit==1.50.0",
    "pyarrow==20.0.0",
    "imageio==2.37.0",
    "imageio-ffmpeg==0.6.0",
]

AVATAR_REQUIREMENTS = [
    "scikit-learn==1.6.1",
    "scikit-image==0.25.2",
    "scipy==1.15.3",
    "soundfile==0.13.1",
    "soxr==0.5.0.post1",
    "librosa==0.11.0",
    "sympy==1.13.1",
    "audio-separator==0.30.2",
    "pyloudnorm==0.1.1",
    "nvidia-ml-py==13.580.65",
    "tzdata==2025.2",
    "onnx==1.18.0",
    "onnxruntime==1.16.3",
    "tritonserverclient==0.0.6",
    "openai==1.75.0",
    "cffi==2.0.0",
    "chardet==5.2.0",
]


def run_install(command, label, project_dir):
    print(f"\n==> Installing {label}", flush=True)
    try:
        subprocess.run(command, cwd=project_dir, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"\nDependency stage failed: {label} (exit code {error.returncode}). "
            "Fix the reported error and rerun the same command; completed "
            "stages will remain installed."
        ) from error


def install_requirements(options):
    project_dir = Path(options.project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise SystemExit(f"LongCat project directory does not exist: {project_dir}")

    pip = [
        sys.executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--retries",
        str(options.retries),
        "--resume-retries",
        str(options.resume_retries),
        "--timeout",
        "120",
    ]
    run_install(
        pip
        + [
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "ninja",
            "packaging",
            "psutil",
        ],
        "packaging and build tools",
        project_dir,
    )
    if options.skip_torch:
        run_install(
            [
                sys.executable,
                "-c",
                "import torch; assert torch.__version__.startswith('2.6.'), torch.__version__; "
                "assert torch.version.cuda.startswith('12.4'), torch.version.cuda",
            ],
            "preinstalled PyTorch",
            project_dir,
        )
    else:
        torch_index = (
            PYPI_INDEX if options.cuda == "cu124" else PYTORCH_INDEXES[options.cuda]
        )
        run_install(
            pip
            + [
                "install",
                "--index-url",
                torch_index,
                *(["--no-deps"] if options.use_system_cuda else []),
                "torch==2.6.0",
                "torchvision==0.21.0",
                "torchaudio==2.6.0",
            ],
            f"PyTorch ({options.cuda})",
            project_dir,
        )
        if options.use_system_cuda:
            run_install(
                pip + ["install", "--prefer-binary"] + TORCH_RUNTIME_REQUIREMENTS,
                "PyTorch Python runtime dependencies",
                project_dir,
            )
    run_install(
        pip + ["install", "--prefer-binary"] + BASE_REQUIREMENTS,
        "base Python requirements",
        project_dir,
    )
    if not options.skip_flash_attn:
        run_install(
            pip + ["install", "flash-attn==2.7.4.post1", "--no-build-isolation"],
            "FlashAttention",
            project_dir,
        )
    if options.avatar:
        run_install(
            pip + ["install", "--prefer-binary"] + AVATAR_REQUIREMENTS,
            "avatar Python requirements",
            project_dir,
        )
        print(
            "Avatar dependencies installed. Install system ffmpeg and libsndfile "
            "as described in README.md."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install_requirements"])
    parser.add_argument(
        "--project-dir",
        default="/workspace/LongCat-Video",
        help="LongCat checkout used as the pip working directory",
    )
    parser.add_argument(
        "--cuda",
        choices=sorted(PYTORCH_INDEXES),
        default="cu124",
        help="PyTorch CUDA wheel channel (default: cu124)",
    )
    parser.add_argument("--avatar", action="store_true")
    parser.add_argument("--skip-flash-attn", action="store_true")
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="use and verify PyTorch already installed in the base environment",
    )
    parser.add_argument(
        "--use-system-cuda",
        action="store_true",
        help="do not install NVIDIA wheels; use CUDA libraries from the base image",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--resume-retries", type=int, default=20)
    options = parser.parse_args()
    if options.retries < 0:
        parser.error("--retries must be zero or greater")
    if options.resume_retries < 0:
        parser.error("--resume-retries must be zero or greater")
    return options


if __name__ == "__main__":
    selected_options = parse_args()
    install_requirements(selected_options)
