# ComfyUI Face Swap on RunPod

This repository contains a ComfyUI workflow for replacing the face in a single-person talking-head video while preserving the original body, background, lighting, and audio.

The workflow is [FaceSwap_HQ_L4_24GB.json](FaceSwap_HQ_L4_24GB.json). It uses ComfyUI, VideoHelperSuite, FaceTools, ReActor, and FaceFilter.

## RunPod ComfyUI template setup

Use an NVIDIA Pod with a Network Volume mounted at `/workspace`. Expose TCP port `8188` through RunPod. The ComfyUI template already provides ComfyUI, Python, PyTorch, and the NVIDIA runtime, so do not clone a second ComfyUI installation, create a second virtual environment, or reinstall PyTorch.

### 1. Clone this repository

Run these commands once in the Pod terminal:

```bash
cd /workspace
if [ ! -d /workspace/facefusion_faceswap/.git ]; then
  git clone --depth 1 https://github.com/pjain-github/facefusion_app.git /workspace/facefusion_faceswap
fi
cd /workspace/facefusion_faceswap
```

The repository name still contains `facefusion`, but FaceFusion is not used.

### 2. Verify the template Python and ComfyUI root

Use the same Python executable that starts ComfyUI:

```bash
which python
python --version
python -c "import sys, torch; print(sys.executable); print('CUDA:', torch.cuda.is_available())"
```

The CUDA check should print `CUDA: True`. Set `COMFYUI` to the actual ComfyUI directory used by the template:

```bash
export COMFYUI=/workspace/ComfyUI
test -f "$COMFYUI/main.py"
```

If the template uses another location, change `COMFYUI` before running the remaining commands.

### 3. Install the custom nodes

The workflow needs these nodes:

- `ComfyUI-VideoHelperSuite`
- `ComfyUI-FaceFilter`
- `comfyui_facetools` or `ComfyUI FaceTools`
- `ComfyUI-ReActor`

The preferred method is **Manager -> Custom Nodes Manager** in the ComfyUI interface. Search for and install each node, then restart ComfyUI.

If Manager is unavailable, install them manually:

```bash
mkdir -p "$COMFYUI/custom_nodes"
cd "$COMFYUI/custom_nodes"
[ -d ComfyUI-VideoHelperSuite ] || git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
[ -d ComfyUI-FaceFilter ] || git clone --depth 1 https://github.com/Kidev/ComfyUI-FaceFilter.git
[ -d comfyui_facetools ] || git clone --depth 1 https://github.com/dchatel/comfyui_facetools.git
[ -d ComfyUI-ReActor ] || git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git
```

Install the repository's additional dependencies with the same Python used by ComfyUI. Do not reinstall PyTorch:

```bash
cd /workspace/facefusion_faceswap
python -m pip install -r requirements.txt
```

Install node-specific requirements only when a file exists or the ComfyUI startup log reports a missing module:

```bash
for requirements in \
  "$COMFYUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt" \
  "$COMFYUI/custom_nodes/ComfyUI-FaceFilter/requirements.txt" \
  "$COMFYUI/custom_nodes/ComfyUI-ReActor/requirements.txt"; do
  if [ -f "$requirements" ]; then python -m pip install -r "$requirements"; fi
done
```

Restart ComfyUI after installing nodes or packages. Use the template's normal startup command; if you need to start it manually, use:

```bash
cd "$COMFYUI"
python main.py --listen 0.0.0.0 --port 8188
```

## Models

Create the model directories first:

```bash
mkdir -p /workspace/ComfyUI/input /workspace/ComfyUI/output
mkdir -p /workspace/ComfyUI/models/insightface/models/antelopev2
mkdir -p /workspace/ComfyUI/models/facerestore_models
mkdir -p /workspace/ComfyUI/models/landmarks
mkdir -p /workspace/ComfyUI/models/bisenet
```

### ReActor model files

These are model files, not Python packages. Download them from the official ReActor dataset and place them at the exact paths below:

The names `swap_model` and `face_restore_model` are ReActor node input names, not directory names. Do not rename these folders to `swap_model` or `face_restore_models`; ReActor expects the directory name `facerestore_models`.

```text
/workspace/ComfyUI/models/insightface/inswapper_128.onnx
/workspace/ComfyUI/models/facerestore_models/GPEN-BFR-1024.onnx
```

RunPod may not have `wget`, and Hugging Face redirects large files to a storage backend. Use `curl -L` so the redirect is followed:

```bash
curl -L --fail --retry 3 \
  -o /workspace/ComfyUI/models/insightface/inswapper_128.onnx \
  'https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx?download=true'

curl -L --fail --retry 3 \
  -o /workspace/ComfyUI/models/facerestore_models/GPEN-BFR-1024.onnx \
  'https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx?download=true'

test -s /workspace/ComfyUI/models/insightface/inswapper_128.onnx
test -s /workspace/ComfyUI/models/facerestore_models/GPEN-BFR-1024.onnx
ls -lh \
  /workspace/ComfyUI/models/insightface/inswapper_128.onnx \
  /workspace/ComfyUI/models/facerestore_models/GPEN-BFR-1024.onnx
```

If a model still does not appear, verify that the running ComfyUI process uses this same root and that ReActor sees the files directly in these folders:

```bash
find /workspace/ComfyUI/models -maxdepth 2 -type f \
  \( -name 'inswapper_128.onnx' -o -name 'GPEN-BFR-1024.onnx' \) -ls
```

Expected approximate sizes are 554 MB for `inswapper_128.onnx` and 285 MB for `GPEN-BFR-1024.onnx`. If either file is only a few kilobytes, it is an error response rather than a model and should be deleted and downloaded again.

ReActor may also download `buffalo_l` on its first use. If it does not, install it through ComfyUI Manager or place the model pack under the ReActor/InsightFace model directory shown in the ReActor startup log.

### FaceFilter model

FaceFilter is configured in this workflow to use the `antelopev2` InsightFace pack. Install it through ComfyUI Manager, or download the `antelopev2` pack from its official model source and extract the pack so its files are directly inside:

```text
/workspace/ComfyUI/models/insightface/models/antelopev2/
```

Do not create an extra nested directory such as `antelopev2/antelopev2/`. The FaceFilter node must list `antelopev2` after ComfyUI is restarted.

### FaceTools models

`CropFaces` is configured to use `BiSeNet`. FaceTools needs the FaceAlignment landmark model and the BiSeNet model. Follow the model links in the FaceTools repository README and place them here:

```text
/workspace/ComfyUI/models/landmarks/
/workspace/ComfyUI/models/bisenet/
```

The exact filenames can vary by FaceTools revision; keep the filenames supplied by the model download instructions. Restart ComfyUI after adding models.

## Load and run the workflow

Upload the source face image and input video through the ComfyUI web interface, or copy them into `/workspace/ComfyUI/input/`. ComfyUI writes generated videos to `/workspace/ComfyUI/output/`.

1. Open the ComfyUI URL on RunPod.
2. Drag [FaceSwap_HQ_L4_24GB.json](FaceSwap_HQ_L4_24GB.json) into ComfyUI.
3. Select a clear source face in `LoadImage`.
4. Select the input video in `VHS_LoadVideo`.
5. Set `VHS_VideoCombine.frame_rate` to the input video FPS. The workflow currently assumes `30` FPS.
6. Queue a short test clip before rendering the complete video.

The original video frames enter `WarpFacesBack`, so the output retains the original background and body. Only the detected face crop is replaced. The original audio is passed through to the output video.

## Single-face recommendation

For a video containing only you, `FaceFilterNode` is optional. The current workflow keeps it so identity filtering can reject false detections, but it can produce a black placeholder when the face is not recognized. If that causes black face patches during head turns or occlusion, connect `CropFaces.crops` directly to `ReActorFaceSwap.input_image` and bypass FaceFilter.

Keep `VHS_BatchManager.frames_per_batch` at `1` for a 24 GB GPU while testing. Increase it only after confirming that VRAM remains available. If the face edge is too hard, increase the four `FeatherMask` values from `32` to `64`; if the face looks excessively soft, reduce them.

## Troubleshooting

- **Custom node not found:** confirm the repository exists under `/workspace/ComfyUI/custom_nodes`, then restart ComfyUI.
- **Model not found:** check the directory and filename against the paths above. Models must not be one directory deeper than shown.
- **Black face patch:** bypass FaceFilter or lower its threshold only after checking the `DEBUG` output.
- **Audio or duration is wrong:** make the output frame rate match the input FPS. Do not force `30` FPS for a 24 or 60 FPS source unless that conversion is intentional.
- **Out of memory:** keep `frames_per_batch` at `1`, reduce `CropFaces` from `768` to `512`, or select a smaller face-restoration model.
- **Video does not play everywhere:** H.265 10-bit is efficient but has limited compatibility. Use H.264 with `yuv420p` for broadly compatible output.

Only process media for which you have the necessary rights and consent. If a real person's face is used, follow applicable law and disclose synthetic alterations where appropriate.
