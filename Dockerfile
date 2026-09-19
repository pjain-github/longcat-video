FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libsndfile1 \
    openssh-server \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/LongCat-Video

RUN git clone --single-branch --branch main \
    https://github.com/meituan-longcat/LongCat-Video .

COPY setup.py /tmp/longcat-setup.py

RUN python3.10 -m venv .venv \
    && .venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && .venv/bin/python /tmp/longcat-setup.py install_requirements \
        --avatar --use-system-cuda --project-dir /opt/LongCat-Video \
    && .venv/bin/python -m pip install jupyterlab onnxruntime-gpu accelerate

# This is the working memory-efficient INT8 loader from the validated pod.
# Do not alter the upstream distilled-avatar guidance logic: Avatar 1.5 sets
# both text and audio guidance to 1.0 when --use_distill is enabled.
COPY docker/quantization.py longcat_video/modules/quantization.py
COPY download_weights.py /opt/LongCat-Video/download_weights.py
COPY docker/entrypoint.sh /usr/local/bin/longcat-entrypoint.sh

RUN mkdir -p /run/sshd \
    && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && ssh-keygen -A \
    && chmod 755 /usr/local/bin/longcat-entrypoint.sh \
    && .venv/bin/python -m py_compile \
        run_demo_avatar_single_audio_to_video.py \
        longcat_video/modules/quantization.py \
        download_weights.py

VOLUME ["/workspace"]

EXPOSE 22 8888

ENTRYPOINT ["/usr/local/bin/longcat-entrypoint.sh"]
CMD []
