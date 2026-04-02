"""MuseTalk API — generate lip-sync video from audio + face image.

Accepts WAV/MP3 audio and a face image via multipart upload,
returns an MP4 video with synchronized lip movements (v1.5).

Models are loaded once at startup and reused across requests.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path
from time import time

import cv2
import numpy as np
import structlog
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

# Add MuseTalk to path
_MUSETALK_DIR = "/app/MuseTalk"
sys.path.insert(0, _MUSETALK_DIR)

from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import datagen

logger = structlog.get_logger(__name__)

_RESULT_DIR = os.getenv("RESULT_DIR", os.path.join(_MUSETALK_DIR, "results"))
_BBOX_SHIFT = int(os.getenv("BBOX_SHIFT", "0"))
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="MuseTalk API", version="0.2.0")

# ---------------------------------------------------------------------------
# Global model references (loaded once at startup)
# ---------------------------------------------------------------------------

_vae = None
_unet = None
_pe = None
_timesteps = None
_audio_processor = None
_face_parsing = None
_models_ready = False


@app.on_event("startup")
async def _load_models() -> None:
    """Preload all MuseTalk models into GPU at startup."""
    global _vae, _unet, _pe, _timesteps, _audio_processor, _face_parsing, _models_ready
    try:
        logger.info("musetalk.loading_models")

        _vae, _unet, _pe = load_all_model(device=_DEVICE)

        # Convert to float16 for faster inference + lower VRAM
        if _DEVICE.type == "cuda":
            _pe = _pe.half()
            _vae.vae = _vae.vae.half()
            _unet.model = _unet.model.half()

        _timesteps = torch.tensor([0], device=_DEVICE)

        from musetalk.whisper.audio2feature import Audio2Feature
        _audio_processor = Audio2Feature(
            model_path=os.path.join("models", "whisper", "tiny.pt"),
        )

        from musetalk.utils.face_parsing import FaceParsing
        _face_parsing = FaceParsing()

        _models_ready = True
        logger.info("musetalk.models_ready")

    except Exception:
        logger.exception("musetalk.model_load_failed")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/generate")
async def generate_video(
    audio: UploadFile = File(..., description="WAV or MP3 audio file"),
    image: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    bbox_shift: int = Form(_BBOX_SHIFT),
    fps: int = Form(15),
    extra_margin: int = Form(10),
    parsing_mode: str = Form("raw"),
) -> FileResponse:
    """Generate a lip-sync video from audio + face image."""
    if not _models_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet",
        )

    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(_RESULT_DIR) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_ext = Path(audio.filename or "input.wav").suffix or ".wav"
    image_ext = Path(image.filename or "input.jpg").suffix or ".jpg"
    audio_path = work_dir / f"input{audio_ext}"
    image_path = work_dir / f"input{image_ext}"

    audio_path.write_bytes(await audio.read())
    image_path.write_bytes(await image.read())

    logger.info(
        "musetalk.generate_start",
        job_id=job_id,
        audio_size=audio_path.stat().st_size,
        image_size=image_path.stat().st_size,
    )

    try:
        t0 = time()
        output_path = _run_pipeline(
            str(image_path),
            str(audio_path),
            str(work_dir),
            bbox_shift=bbox_shift,
            fps=fps,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
        )
        elapsed = round(time() - t0, 1)

        logger.info(
            "musetalk.generate_done",
            job_id=job_id,
            elapsed_sec=elapsed,
            video_path=output_path,
        )

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    except Exception as exc:
        logger.exception("musetalk.generate_error", job_id=job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@app.get("/health")
async def health_check() -> dict[str, str]:
    if not _models_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models loading",
        )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Pipeline (runs in-process, reuses preloaded models)
# ---------------------------------------------------------------------------


def _run_pipeline(
    image_path: str,
    audio_path: str,
    work_dir: str,
    *,
    bbox_shift: int = 0,
    fps: int = 25,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    batch_size: int = 8,
) -> str:
    """Run MuseTalk inference. Returns path to output MP4."""
    # 1. Detect face landmarks and bounding box
    # get_landmark_and_bbox expects a list of image file paths
    coord_list, frame_list = get_landmark_and_bbox([image_path], bbox_shift)

    # 2. Encode frames to latents
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
        latents = _vae.get_latents_for_unet(crop)
        input_latent_list.append(latents)

    if not input_latent_list:
        raise ValueError("No face detected in image")

    # 3. Extract audio features
    whisper_feature = _audio_processor.audio2feat(audio_path)
    whisper_chunks = _audio_processor.feature2chunks(
        whisper_feature,
        fps=fps,
    )

    # 4. Cycle frames for multi-frame sources
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # 5. Batch inference
    # Convert whisper chunks from numpy to tensors
    whisper_chunks_tensor = [torch.from_numpy(c).to(device=_DEVICE, dtype=torch.float16) if isinstance(c, np.ndarray) else c for c in whisper_chunks]

    gen = datagen(
        whisper_chunks=whisper_chunks_tensor,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
    )

    res_frame_list = []
    for whisper_batch, latent_batch in gen:
        whisper_batch = whisper_batch.to(device=_DEVICE, dtype=torch.float16)
        latent_batch = latent_batch.to(device=_DEVICE, dtype=torch.float16)
        audio_feat = _pe(whisper_batch)
        pred = _unet.model(
            latent_batch, _timesteps, encoder_hidden_states=audio_feat,
        ).sample
        recon = _vae.decode_latents(pred)
        for frame in recon:
            res_frame_list.append(frame)

    # 6. Blend results and write video directly (skip intermediate PNGs)
    n = len(res_frame_list)
    coords_extended = (coord_list_cycle * (n // len(coord_list_cycle) + 1))[:n]
    frames_extended = (frame_list_cycle * (n // len(frame_list_cycle) + 1))[:n]

    first_frame = frames_extended[0]
    h, w = first_frame.shape[:2]

    temp_vid = os.path.join(work_dir, "temp.mp4")
    output_vid = os.path.join(work_dir, "output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_vid, fourcc, fps, (w, h))

    frame_count = 0
    for res_frame, bbox, ori_frame in zip(res_frame_list, coords_extended, frames_extended):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        combine = get_image(
            ori_frame, res_frame, bbox,
            mode=parsing_mode,
            fp=_face_parsing,
        )
        writer.write(combine)
        frame_count += 1

    writer.release()
    logger.info("musetalk.frames_written", count=frame_count)

    # Merge audio
    r = subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", audio_path, "-i", temp_vid,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_vid,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("musetalk.ffmpeg_merge_failed", stderr=r.stderr)
        raise RuntimeError(f"ffmpeg merge failed: {r.stderr}")

    return output_vid
