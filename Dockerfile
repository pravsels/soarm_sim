# CUDA toolkit needed at build time for madrona_mjx
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/root/.local/bin:${CUDA_HOME}/bin:${PATH}"
ENV MUJOCO_GL=egl

# System deps, newer CMake, Python 3.11 + pip, Vulkan loader (dev adds libvulkan.so),
# and uv – all in one layer
RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    software-properties-common curl git ca-certificates wget gpg \
    build-essential libegl1 libgles2 libx11-6 libxext6 \
  ; \
  wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg; \
  echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
    > /etc/apt/sources.list.d/kitware.list; \
  add-apt-repository ppa:deadsnakes/ppa -y; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    cmake python3.11 python3.11-distutils python3.11-dev \
    libvulkan1 libvulkan-dev vulkan-tools \
  ; \
  curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11; \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1; \
  python -m pip install --no-cache-dir --upgrade pip; \
  curl -LsSf https://astral.sh/uv/install.sh | sh; \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Build & install madrona_mjx (Vulkan dev already provides libvulkan.so → no manual symlink)
RUN set -eux; \
  git clone https://github.com/shacklettbp/madrona_mjx.git /opt/madrona_mjx; \
  cd /opt/madrona_mjx; \
  git submodule update --init --recursive; \
  cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda; \
  cmake --build build -j"$(nproc)"; \
  pip install --no-cache-dir -e .

CMD ["bash"]
