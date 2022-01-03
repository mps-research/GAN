# FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Tokyo apt-get install -y \
        build-essential \
        libssl-dev \
        libffi-dev \ 
        python3-dev \
        python3-pip \
        libtiff5-dev \
        libjpeg8-dev \
        libopenjp2-7-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        tcl8.6-dev \
        tk8.6-dev \ 
        python3-tk \
        libharfbuzz-dev \
        libfribidi-dev \
        libxcb1-dev

RUN pip3 install \
        Pillow \
        tqdm \
        torch \
        torchvision \
        torchaudio \
        "ray[tune]" \
        tensorboard

COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ubuntu && \
    useradd -g ${GID} -u ${UID} -m -s /bin/bash ubuntu

RUN mkdir /code /logs /data /models
COPY ./src /code
RUN chown -R ${UID}:${GID} /code /logs /data /models

WORKDIR /code

USER ubuntu

ENTRYPOINT ["entrypoint.sh"]

