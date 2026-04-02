# MuseTalk API

REST API for real-time lip-sync video generation from audio + face image, powered by [MuseTalk](https://github.com/TMElyralab/MuseTalk).

## API

### `POST /generate`

Generate a lip-sync video from audio + face image.

**Request** (multipart/form-data):
- `audio` — WAV or MP3 file
- `image` — Face image (JPEG/PNG)
- `bbox_shift` — int, default 0 (positive = more mouth openness)

**Response**: MP4 video file (256x256 face region)

```bash
curl -X POST http://localhost:8008/generate \
  -F "audio=@voice.wav" \
  -F "image=@avatar.jpg" \
  -o output.mp4
```

### `GET /health`

Returns `{"status": "ok"}`.

## Setup

### 1. Download models

```bash
mkdir -p models/musetalk models/dwpose models/face-parse-bisent models/sd-vae-ft-mse models/whisper

# MuseTalk weights
huggingface-cli download TMElyralab/MuseTalk --local-dir models/musetalk

# SD VAE
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae-ft-mse

# Whisper tiny
python -c "import whisper; whisper.load_model('tiny', download_root='models/whisper')"

# DWPose + face-parse: see MuseTalk docs
```

### 2. Run

```bash
docker compose up --build
```

### 3. Test

```bash
curl -X POST http://localhost:8008/generate \
  -F "audio=@test.wav" \
  -F "image=@face.jpg" \
  -o result.mp4
```

## Requirements

- NVIDIA GPU with ~2-4 GB VRAM
- nvidia-container-toolkit
- Docker with GPU support

## Credits

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) — TMElyralab (Tencent Music)
