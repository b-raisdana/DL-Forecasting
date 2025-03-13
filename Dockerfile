# docker run --gpus all -it --rm -v docker_volume:/workspace nvcr.io/nvidia/tensorflow:25.01-tf2-py3
# Use TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Install NVIDIA CUDA Toolkit and cuDNN at build time
RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-4 \
    libcudnn8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set LD_LIBRARY_PATH permanently
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# Set TensorFlow GPU options
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=0

# Fix missing CUDA library symbolic links (only create if missing)
RUN /bin/sh -c "[ ! -L /usr/lib/x86_64-linux-gnu/libcudnn.so ] && ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so || true" && \
    /bin/sh -c "[ ! -L /usr/local/cuda/lib64/libcudart.so ] && ln -s /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so || true" && \
    /bin/sh -c "[ ! -L /usr/local/cuda/lib64/libcusolver.so ] && ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so || true" && \
    /bin/sh -c "[ ! -L /usr/local/cuda/lib64/libcublas.so ] && ln -s /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so || true" && \
    /bin/sh -c "[ ! -L /usr/local/cuda/lib64/libcufft.so ] && ln -s /usr/local/cuda/lib64/libcufft.so.11 /usr/local/cuda/lib64/libcufft.so || true" && \
    /bin/sh -c "[ ! -L /usr/local/cuda/lib64/libcusparse.so ] && ln -s /usr/local/cuda/lib64/libcusparse.so.12 /usr/local/cuda/lib64/libcusparse.so || true"

# Set working directory inside the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . /app

# Install additional dependencies (if needed)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set default command to run Python scripts
ENTRYPOINT ["python3"]
