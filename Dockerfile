# CUDA 12.4.1 + cuDNN 開發版 Base Image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgl1 \
    && apt-get clean


# 建立虛擬環境
RUN python3.12 -m venv /venv

# 設定 PATH，使用虛擬環境的 pip 與 python
ENV PATH="/venv/bin:$PATH"

# 升級 pip
RUN pip install --upgrade pip

# 複製通用套件清單並安裝（從 PyPI）
COPY requirements.txt .
RUN pip install -r requirements.txt

# 安裝 PyTorch + CUDA 12.8 專屬版本（從 PyTorch 官方）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# 設定工作目錄並複製程式
WORKDIR /app
COPY . /app

# 預設進入 bash
CMD ["python", "main.py"]

