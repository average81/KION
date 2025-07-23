FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install -y tzdata software-properties-common cmake build-essential libboost-all-dev libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libgl1 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt
RUN python3.9 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN python3.9 -m pip install numpy==1.24.3 pandas==1.5.3 --force-reinstall

RUN rm -f /usr/local/bin/cmake
RUN ln -s /usr/bin/cmake /usr/local/bin/cmake
RUN python3.9 -m pip install dlib

RUN python3.9 -m pip cache purge

COPY . .

# Этап 2: финальный минимальный образ
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3-pip \
        libopenblas0 liblapack3 libx11-6 libgtk-3-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --upgrade pip

WORKDIR /app

COPY . .

# Копируем все библиотеки из builder
COPY --from=builder /usr/local /usr/local

CMD ["python3.9", "shorts.py", "--help"]