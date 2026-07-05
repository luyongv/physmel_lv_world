FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt requirements.txt
RUN python3.10 -m pip install --upgrade pip==23.3.2 && python3.10 -m pip install -r requirements.txt
COPY . .
RUN python3.10 -m pip install -e .
CMD ["python3.10", "-m", "physmel_lv_world.gateways.train", "settings/longhaul/main.yaml"]
