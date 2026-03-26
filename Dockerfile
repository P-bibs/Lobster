FROM nvcr.io/nvidia/cuda:12.5.0-devel-ubuntu20.04
#FROM ubuntu:focal

# set environment variables for tzdata
ARG TZ=America/New_York
ENV TZ=${TZ}

# include manual pages and documentation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
  yes | unminimize

# install GCC-related packages
RUN apt-get -y install\
 binutils-doc\
 cpp-doc\
 gcc-doc\
 g++\
 g++-multilib\
 gdb\
 gdb-doc\
 glibc-doc\
 libblas-dev\
 liblapack-dev\
 liblapack-doc\
 libstdc++-10-doc\
 make\
 make-doc

# install clang-related packages
RUN apt-get -y install\
 clang\
 lldb\
 clang-format

# install cmake packages
RUN apt-get -y install\
  cmake

# install programs used for system exploration
RUN apt-get -y install\
 blktrace\
 linux-tools-generic\
 nsight-systems\
 strace

# install interactive programs (emacs, vim, nano, man, sudo, etc.)
RUN apt-get -y install\
 bc\
 zsh\
 curl\
 dc\
 git\
 git-doc\
 man\
 micro\
 nano\
 neovim\
 psmisc\
 sudo\
 wget\
 zip\
 unzip\
 tar

# python
RUN apt-get -y install\
  python3\
  python3-pip\
  python3-venv

# set up libraries
RUN apt-get -y install\
 locales
# libreadline-dev\
# wamerican

# install programs used for networking
RUN apt-get -y install\
 time

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y

RUN apt-get -y install dwarves

RUN mkdir /root/Builds && cd /root/Builds\
  && wget https://github.com/Kitware/CMake/releases/download/v3.31.0-rc2/cmake-3.31.0-rc2-linux-x86_64.tar.gz \
  && tar -xvf cmake-3.31.0-rc2-linux-x86_64.tar.gz\
  && rm cmake-3.31.0-rc2-linux-x86_64.tar.gz
RUN mkdir -p /root/.local/bin && ln -s /root/Builds/cmake-3.31.0-rc2-linux-x86_64/bin/cmake /root/.local/bin/cmake
ENV PATH="/root/.local/bin:$PATH"

RUN python3 -m venv /root/venv &&\
/root/venv/bin/pip install --upgrade pip &&\
/root/venv/bin/pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124 &&\
/root/venv/bin/pip install maturin transformers gym scikit-learn opencv-python tqdm matplotlib wandb

# set up default locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# remove unneeded .deb files
RUN rm -r /var/lib/apt/lists/*

# set up passwordless sudo for user paulbib
# RUN useradd -rm -d /home/paulbib -s /bin/zsh -g root -G sudo -u 1021 paulbib && \
#   echo "paulbib ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/paulbib-init

# create binary reporting version of dockerfile
# RUN (echo '#\!/bin/sh'; echo 'echo 1') > /usr/bin/docker-version; chmod ugo+rx,u+w,go-w /usr/bin/docker-version

# configure your environment
# USER paulbib
# RUN rm -f ~/.bash_logout

# WORKDIR /home/paulbib
CMD ["/bin/zsh", "-l"]

