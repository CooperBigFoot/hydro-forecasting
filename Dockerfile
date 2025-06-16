# Use RunPod's PyTorch base image with CUDA support and Python 3.11
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Configure uv environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.11

# Copy dependency files for caching
COPY pyproject.toml uv.lock /workspace/

# Install dependencies (without the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Configure Git (for your workflow)
RUN git config --global user.name "CooperBigFoot" && \
    git config --global user.email "nlazaro@student.ethz.ch"

# Keep the default RunPod startup command
CMD ["/start.sh"]