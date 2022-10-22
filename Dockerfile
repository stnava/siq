FROM python:3.7-slim-buster
LABEL maintainer="stnava"

RUN apt-get update && \
    apt-get install -y build-essential cmake libpng-dev pkg-config git

RUN pip install numpy keras boto3
RUN pip install --upgrade tensorflow tensorflow-probability

RUN pip install antspyx antspynet antspyt1w

RUN python3 setup.py install
