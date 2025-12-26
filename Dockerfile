# syntax=docker/dockerfile:1

# Based on the official NVIDIA-certified image 25.02
# This has TF 2.17, Python 3.12, CUDA 12.8.0, cuDNN 9.7.1
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /app
ENV PYTHONPATH="/app"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements_training.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]