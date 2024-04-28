ARG BASE_TAG=24.03-py3
FROM nvcr.io/nvidia/pytorch:$BASE_TAG


COPY . /workspace/MN5-Distributed-PyTorch

RUN pip install -r /workspace/MN5-Distributed-PyTorch/singularity/requirements.txt

# Install flash-attn
RUN pip install flash-attn==2.5.5 --no-build-isolation

# helper for huggingface-login cli
RUN git config --global credential.helper store