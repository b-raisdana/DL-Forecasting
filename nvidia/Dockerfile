# Base Image with CUDA 12.2 and Ubuntu 22.04
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    nano \
    python3 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA and GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# Ensure CUDA is correctly linked
RUN ln -s /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so || true

# Set working directory
WORKDIR /workspace

# Default command to verify GPU
CMD ["nvidia-smi"]
