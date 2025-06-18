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

# Install and configure SSH daemon
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    # Configure SSH daemon
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#AuthorizedKeysFile/AuthorizedKeysFile/' /etc/ssh/sshd_config && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Expose SSH port
EXPOSE 22

# Create startup script that starts SSH daemon and then the default RunPod command
RUN echo '#!/bin/bash\n\
    # Start SSH daemon\n\
    service ssh start\n\
    # Run the original RunPod startup command\n\
    exec /start.sh "$@"' > /startup.sh && \
    chmod +x /startup.sh

# Use the new startup script
CMD ["/startup.sh"]