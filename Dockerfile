FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
ENV CONDA_ACCEPT_LICENSE=yes
ENV DLIB_USE_CUDA=1

RUN apt-get update && \
    apt-get install -y cmake build-essential libboost-all-dev libopenblas-dev liblapack-dev wget software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Установка Miniconda (старая версия)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Настройка conda
RUN conda config --set channel_priority strict

# Принудительно устанавливаем Python 3.9
RUN conda install python=3.9 -y

RUN python -m pip install --upgrade pip

WORKDIR /app

# Сначала устанавливаем PyTorch CUDA версию
RUN python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# Устанавливаем dlib через conda-forge
RUN conda install -c conda-forge dlib -y

# Устанавливаем spacy через conda-forge (conda сам выберет версию)
RUN conda install -c conda-forge spacy -y

# Устанавливаем языковую модель для русского языка через spacy
RUN python -m spacy download ru_core_news_md

RUN python -m pip cache purge

COPY . .

# Этап 2: финальный минимальный образ
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ACCEPT_LICENSE=yes
ENV DLIB_USE_CUDA=1

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3-pip libopenblas0 liblapack3 \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем все библиотеки из builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

# Устанавливаем PATH: conda Python будет основным (где установлены библиотеки)
ENV PATH="/opt/conda/bin:$PATH"
# Создаем символическую ссылку python -> conda python
RUN ln -sf /opt/conda/bin/python /usr/bin/python

ENV PYTHON_CMD="python"
CMD ["python", "shorts.py", "--help"]