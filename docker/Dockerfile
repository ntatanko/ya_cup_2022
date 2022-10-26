FROM nvcr.io/nvidia/tensorflow:22.09-tf2-py3
#FROM python:latest

# where external files are mapped to
ARG APP_DIR="/app"

# build directory
WORKDIR /tmp

# supress too many annying messages from tf
ENV TF_CPP_MIN_LOG_LEVEL=3

# for vs code, see https://aka.ms/vscode-docker-python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# use bash as a default shell
ENV SHELL=/bin/bash
ENV force_color_prompt=yes

# path to app modules
ENV PYTHONPATH=${APP_DIR}:${APP_DIR}/lib:${APP_DIR}/src

# apt setup
RUN apt-get update
ENV TZ=America/Los_Angles
ENV DEBIAN_FRONTEND=noninteractive

# node js (required by for jupyterlab_code_formatter and plotly)
RUN apt-get install -y nodejs npm
RUN npm install -g n && n lts

# pip
RUN pip install --upgrade pip

# jupyter lab
RUN pip install jupyterlab==3.4.1

# jupyterlab_code_formatter extension
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter@v1.4.11 --no-build
RUN pip install jupyterlab_code_formatter==v1.4.11
RUN jupyter server extension enable --py jupyterlab_code_formatter
RUN pip install isort yapf black

# plotly jl extension
RUN pip install "ipywidgets>=7.5"
RUN jupyter labextension install jupyterlab-plotly --no-build
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget --no-build
RUN pip install plotly

# lsp jl extension
# https://github.com/krassowski/jupyterlab-lsp
RUN pip install jupyterlab-lsp
RUN pip install python-language-server[all]

# jl debugger extension
# RUN pip install xeus-python
# RUN jupyter labextension install @jupyterlab/debugger --no-build

# build jupyter lab assets
RUN jupyter lab build

# jupyter config saving
RUN echo "cp -r /root/.jupyter ${APP_DIR}/docker/res/root/ && \
    rm -rf ${APP_DIR}/docker/res/root/.jupyter/lab/workspaces && echo \"done\"" \
    >> /usr/bin/save-jupyter-config
RUN chmod +x /usr/bin/save-jupyter-config

# startup commands
ARG JUPYTER_TOKEN="mytoken"
RUN echo "tensorboard --logdir=${APP_DIR}/.tensorboard --bind_all &" >> /cmd
RUN echo "jupyter lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='${JUPYTER_TOKEN}' --NotebookApp.allow_origin='*' &" >> /cmd
RUN echo "sleep infinity" >> /cmd
CMD ["sh", "/cmd"]

# needed for keras model visualization
RUN pip install pydot
RUN apt-get install -y graphviz

# kaggle cli
ENV KAGGLE_CONFIG_DIR ${APP_DIR}
RUN pip install kaggle

# pre-approve github host
RUN mkdir /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

# torch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# for ScaledYOLOv4
# RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# python packages
RUN pip install pandas
RUN pip install --upgrade numpy
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install -U scikit-learn
RUN pip install scipy
RUN pip install albumentations
RUN pip install tqdm
RUN apt-get install -y libsndfile1
RUN pip install tables
RUN pip install tfa-nightly
RUN pip install pillow
RUN pip install onnx


# tmp fix for https://github.com/jupyterlab/jupyterlab/issues/10228
RUN jupyter server extension disable nbclassic

# copy external resources
COPY docker/res /

WORKDIR ${APP_DIR}
