FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .

# Install requirements
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Set the default MuJoCo rendering backend to EGL, which is compatible with headless environments and does not require a display server.
ENV MUJOCO_GL=egl
