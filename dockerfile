FROM nvcr.io/nvidia/pytorch:22.11-py3
ADD requirements.txt ./
RUN pip install -r requirements.txt