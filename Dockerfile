FROM tensorflow/tensorflow:2.13.0-gpu

RUN apt-get update && \
    apt-get install -y libusb-1.0-0 libusb-1.0-0-dev git

RUN pip install tflite-model-maker pycocotools

ARG WKDIR=/workdir
WORKDIR ${WKDIR}