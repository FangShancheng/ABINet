FROM anibali/pytorch:cuda-9.0
MAINTAINER fangshancheng <fangsc@ustc.edu.cn>
RUN sudo rm -rf /etc/apt/sources.list.d && \
    sudo apt update && \
    sudo apt install -y build-essential vim && \
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install torch==1.1.0 torchvision==0.3.0 && \
    pip install fastai==1.0.60 && \
    pip install ipdb jupyter ipython lmdb editdistance tensorboardX natsort nltk && \
    conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo && \
    pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo && \
    conda install -yc conda-forge libjpeg-turbo && \
    CFLAGS="${CFLAGS} -mavx2" pip install --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd==6.2.2.post1 && \
    conda install -y jpeg libtiff opencv && \
    sudo rm -rf /var/lib/apt/lists/* && \
    sudo rm -rf /tmp/* && \
    sudo rm -rf ~/.cache && \
    sudo apt clean all && \
    conda clean -y -a
EXPOSE 8888
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
