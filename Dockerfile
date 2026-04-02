# MuseTalk API — real-time lip-sync video generation
# Base: PyTorch 2.0.1 + CUDA 11.7 (as recommended by MuseTalk)

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV FFMPEG_PATH=/usr/bin/ffmpeg

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git libgl1 libglib2.0-0 gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone MuseTalk
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

WORKDIR /app/MuseTalk

# Install MuseTalk dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Install API dependencies
RUN pip install --no-cache-dir \
        fastapi[standard] \
        structlog \
        python-multipart

# Create results directory
RUN mkdir -p /app/results

WORKDIR /app/MuseTalk

# Copy API server
COPY main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
