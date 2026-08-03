#!/usr/bin/env bash
set -euo pipefail

cd /opt/LongCat-Video

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
    --ServerApp.preferred_dir=/workspace \
    --IdentityProvider.token="${JUPYTER_TOKEN}"