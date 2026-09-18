# LongCat-Video Avatar 1.5 on RunPod

This runbook describes the validated Avatar 1.5 INT8 setup on an NVIDIA A40.
The tested configuration produced a 93-frame, 3.72-second video successfully.

## Validated configuration

- GPU: NVIDIA A40, 48 GB VRAM
- System RAM: 64 GB or more recommended
- CUDA: 12.4
- Manual pod Python: 3.10
- Docker base: CUDA 12.4 with cuDNN development libraries
- Docker Python: 3.10 with PyTorch 2.6.0+cu124
- FlashAttention: 2.7.4.post1
- Model: Avatar 1.5 INT8 with DMD LoRA

The model files use about 42 GB in total:

- Avatar 1.5 files: about 21 GB
- Foundational LongCat files: about 22 GB

Keep the weights on a RunPod Network Volume. Do not bake them into the Docker
image. The image stays smaller, and future pods can reuse the same volume.

## Recommended deployment

Use the Docker image and RunPod template files in this repository:

- `Dockerfile` uses the CUDA development image and a project-local `.venv`.
- `docker/quantization.py` is the working memory-efficient INT8 loader.
- `docker/entrypoint.sh` starts Jupyter and SSH, or runs inference commands.
- `download_weights.py` downloads only the model files required by Avatar 1.5.
- `runpod-template.json` contains the template settings.

### Build and push with GitHub Actions

The repository includes `.github/workflows/publish-image.yml`. GitHub Actions
builds the image on an x86_64 runner and pushes it directly to GHCR, so Docker
Desktop and local disk space are not required.

1. Push this repository to GitHub.
2. Open the repository's **Actions** tab.
3. Select **Publish LongCat image**.
4. Choose **Run workflow** on the desired branch.

The workflow publishes:

```text
ghcr.io/pjain-github/longcat-video:avatar-1.5-fast
```

The workflow uses the built-in `GITHUB_TOKEN` with package write permission.
After the first successful run, open the package's settings under the GitHub
profile and set its visibility to **Public**. A public image can be pulled by
RunPod without extra registry credentials. Keep the package private only if you
configure RunPod with matching GHCR image-pull credentials.

The Docker build installs the PyTorch 2.6.0 Python wheels from PyPI with
`--no-deps` and uses the CUDA/cuDNN libraries already present in the base image.
This avoids downloading a second copy of the large NVIDIA wheel stack, which
otherwise exhausts the GitHub-hosted runner's disk during installation.

For a local or other x86_64 builder, the equivalent command is:

```bash
docker login ghcr.io
docker buildx build --platform linux/amd64 \
  --tag ghcr.io/pjain-github/longcat-video:avatar-1.5 \
  --push .
```

### Create the RunPod template

Use `runpod-template.json` or enter these values in the RunPod UI:

- Custom image: `ghcr.io/pjain-github/longcat-video:avatar-1.5-fast`
- GPU: NVIDIA A40 or another GPU with at least 48 GB VRAM
- Container disk: 30 GB
- Network Volume: at least 80 GB
- Volume mount: `/workspace`
- Exposed ports: `8888/http,22/tcp`
- Jupyter: enabled
- SSH: enabled
- `JUPYTER_TOKEN`: set a private value in RunPod
- `SSH_PUBLIC_KEY`: set to the contents of your local public key file if you
  want direct public-IP SSH or SCP/SFTP access.

Do not mount the Network Volume over `/opt/LongCat-Video`; that is where the
image stores the application code. The volume should be mounted at `/workspace`.

### Download weights after startup

Connect to the new pod and run:

```bash
cd /opt/LongCat-Video
.venv/bin/python download_weights.py --destination /workspace/weights
```

The script is resumable and idempotent. It downloads both the Avatar 1.5
checkpoint and the foundational LongCat files, while excluding unused files
such as FP32, PyTorch binary, and Flax checkpoints.

For only the Avatar files:

```bash
.venv/bin/python download_weights.py \
  --destination /workspace/weights \
  --avatar-only
```

The tested avatar workflow needs both Avatar 1.5 and foundational files, so use
the default command unless you already have the foundational model.

Verify the download:

```bash
du -sh /workspace/weights/LongCat-Video-Avatar-1.5
# approximately 21G

du -sh /workspace/weights/LongCat-Video
# approximately 22G
```

## Manual setup on an existing pod

Use this only when running without the Docker image. The working server used a
project-local `.venv` at `/workspace/LongCat-Video/.venv`.

Install system packages:

```bash
apt-get update
apt-get install -y git ffmpeg libsndfile1 tmux
```

Create or activate Python 3.10:

```bash
python3.10 -m venv /workspace/LongCat-Video/.venv
source /workspace/LongCat-Video/.venv/bin/activate
```

Install dependencies from this repository:

```bash
cd /path/to/longcat-video
python setup.py install_requirements \
  --avatar \
  --project-dir /workspace/LongCat-Video
```

The installer uses CUDA 12.4 by default. The validated alternative channels
are `--cuda cu126` and `--cuda cu128`.

Copy the working loader into the LongCat checkout if the checkout is not the
same source tree:

```bash
cp docker/quantization.py \
  /workspace/LongCat-Video/longcat_video/modules/quantization.py
```

Then download the weights with `download_weights.py`, using the persistent
`/workspace/weights` directory.

## Input file

Create a JSON file such as `/workspace/my_inputs/my_video.json`:

```json
{
  "prompt": "Professional technology YouTuber speaking naturally to camera.",
  "cond_image": "/workspace/my_inputs/me.webp",
  "cond_audio": {
    "person1": "/workspace/my_inputs/voice.mp3"
  }
}
```

Prompts must be a single line. WEBP images and MP3 audio are supported.

## Run Avatar 1.5

Always use `torchrun`, not direct Python. From the Docker image:

```bash
cd /opt/LongCat-Video
.venv/bin/torchrun --nproc_per_node=1 \
  run_demo_avatar_single_audio_to_video.py \
  --input_json /workspace/my_inputs/my_video.json \
  --output_dir /workspace/outputs/my_video \
  --resolution 480p \
  --num_segments 1 \
  --stage_1 ai2v \
  --checkpoint_dir /workspace/weights/LongCat-Video-Avatar-1.5 \
  --model_type avatar-v1.5 \
  --use_int8 \
  --use_distill
```

One segment is 93 frames, approximately 3.72 seconds at 25 FPS. Increase
`--num_segments` for longer audio. A six-segment run is approximately 22.3
seconds of generated video.

For long jobs, use tmux:

```bash
tmux new -s longcat
# run the command above
tmux detach -s longcat
tmux attach -t longcat
```

## Verified smoke test

The bundled example was tested successfully with:

```bash
cd /workspace/LongCat-Video
.venv/bin/torchrun --nproc_per_node=1 \
  run_demo_avatar_single_audio_to_video.py \
  --input_json assets/avatar/single_example_1.json \
  --output_dir outputs/avatar-smoke-test \
  --resolution 480p \
  --num_segments 1 \
  --stage_1 ai2v \
  --checkpoint_dir ./weights/LongCat-Video-Avatar-1.5 \
  --model_type avatar-v1.5 \
  --use_int8 \
  --use_distill
```

It produced:

```text
outputs/avatar-smoke-test/ai2v_demo_1.mp4
```

The output was 768x512, 93 frames, 3.72 seconds, with audio.

## Troubleshooting

### `KeyError: RANK`

The demo was started with direct Python. Use `torchrun --nproc_per_node=1`.

### `JSONDecodeError`

The prompt contains a newline or invalid JSON. Keep the prompt on one line and
validate the JSON before running.

### `scheduler_config.json` missing

The scheduler file is included by `download_weights.py`. Do not delete the
`scheduler` directory from either model tree.

### `SIGKILL` or exit code `-9`

The unmodified upstream INT8 loader constructs a large full-precision model
before quantization. This repository's `docker/quantization.py` uses a meta
device, replaces linear layers before allocation, and loads safetensors shards
one at a time. Make sure the working loader is present in the checkout.

### ONNX Runtime CUDA warning

The tested run reported that `CUDAExecutionProvider` was unavailable, so vocal
separation ran on CPU. This is a performance warning; the avatar generation
still completed successfully. The tested run took about two minutes for vocal
separation and about four minutes for one 480p segment.

### Disk quota exceeded

Use a larger Network Volume. Clear only disposable caches if needed:

```bash
pip cache purge
rm -rf /workspace/.cache
```

Do not delete completed model shards to recover from an interrupted download;
rerun `download_weights.py` instead.
