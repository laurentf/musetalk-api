"""MuseTalk API — generate lip-sync video from audio + face image.

Accepts WAV/MP3 audio and a face image via multipart upload,
returns an MP4 video with synchronized lip movements.
"""

import os
import subprocess
import uuid
from pathlib import Path
from time import time

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

logger = structlog.get_logger(__name__)

_RESULT_DIR = os.getenv("RESULT_DIR", "/app/results")
_MUSETALK_DIR = "/app/MuseTalk"
_BBOX_SHIFT = int(os.getenv("BBOX_SHIFT", "0"))

app = FastAPI(title="MuseTalk API", version="0.1.0")


@app.post("/generate")
async def generate_video(
    audio: UploadFile = File(..., description="WAV or MP3 audio file"),
    image: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    bbox_shift: int = _BBOX_SHIFT,
) -> FileResponse:
    """Generate a lip-sync video from audio + face image.

    Returns the MP4 video file directly.
    """
    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(_RESULT_DIR) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write uploaded files to disk
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

    # Write inference config YAML
    config_path = work_dir / "config.yaml"
    output_vid = work_dir / "output.mp4"
    config_path.write_text(
        f"video_path: {image_path}\n"
        f"audio_path: {audio_path}\n"
        f"bbox_shift: {bbox_shift}\n"
        f"result_dir: {work_dir}\n"
    )

    try:
        t0 = time()

        result = subprocess.run(
            [
                "python", "-m", "scripts.inference",
                "--inference_config", str(config_path),
                "--bbox_shift", str(bbox_shift),
            ],
            cwd=_MUSETALK_DIR,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            logger.error(
                "musetalk.inference_failed",
                job_id=job_id,
                stderr=result.stderr[-500:] if result.stderr else "",
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.stderr[-500:] if result.stderr else "Inference failed",
            )

        # Find the output video (MuseTalk names it based on input)
        video_path = _find_output_video(work_dir)
        if not video_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No output video found",
            )

        elapsed = round(time() - t0, 1)
        logger.info(
            "musetalk.generate_done",
            job_id=job_id,
            elapsed_sec=elapsed,
            video_path=str(video_path),
        )

        return FileResponse(
            str(video_path),
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    except subprocess.TimeoutExpired:
        logger.error("musetalk.timeout", job_id=job_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timed out",
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("musetalk.generate_error", job_id=job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


def _find_output_video(work_dir: Path) -> Path | None:
    """Find the generated video in the work directory."""
    # MuseTalk outputs to result_dir with various naming patterns
    for pattern in ["*.mp4", "**/*.mp4"]:
        for f in work_dir.glob(pattern):
            if f.stat().st_size > 1000:  # Skip tiny/empty files
                return f
    return None
