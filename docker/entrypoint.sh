#!/usr/bin/env bash
set -euo pipefail

cd /opt/LongCat-Video

# RunPod can provide a per-Pod public key for direct SSH/SCP access.
if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
    install -d -m 700 /root/.ssh
    touch /root/.ssh/authorized_keys
    if ! grep -qxF "$SSH_PUBLIC_KEY" /root/.ssh/authorized_keys; then
        printf '%s\n' "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
    fi
    chmod 600 /root/.ssh/authorized_keys
fi

/usr/sbin/sshd

if [[ "$#" -gt 0 ]]; then
    case "$1" in
        jupyter)
            shift
            exec .venv/bin/jupyter "$@"
            ;;
        python)
            shift
            exec .venv/bin/python "$@"
            ;;
        torchrun)
            shift
            exec .venv/bin/torchrun "$@"
            ;;
        *)
            exec "$@"
            ;;
    esac
fi

if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
    echo "JUPYTER_TOKEN must be set when starting the default Jupyter service." >&2
    exit 1
fi

exec .venv/bin/jupyter lab \
    --allow-root \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8888 \
    --ServerApp.allow_origin='*' \
    --ServerApp.root_dir=/workspace \
    --IdentityProvider.token="${JUPYTER_TOKEN}"
