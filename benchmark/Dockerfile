FROM nvcr.io/nvidia/cuda:11.7.1-devel-ubuntu20.04

# #install python and pytorch
RUN apt-get clean && apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8 && apt install -y python3 && apt install -y python3-pip
RUN pip install pandas && pip install networkx && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# install nsys
COPY NsightSystems-linux-public-2022.2.1.31-5fe97ab.run /root/NsightSystems-linux-public-2022.2.1.31-5fe97ab.run
COPY pytorch_benchmark /root/pytorch_benchmark
COPY PyProf /root/PyProf

# install nsight
#RUN /root/NsightSystems-linux-public-2022.2.1.31-5fe97ab.run --accept --quiet

# env
ENV PYTHONPATH "${PYTHONPATH}:/root/pytorch_benchmark"
ENV PATH "${PATH}:/usr/local/cuda/bin:/opt/nvidia/nsight-systems/2022.2.1/bin"
ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"

VOLUME /root/data

ENTRYPOINT ["tail", "-f", "/dev/null"]