FROM ubuntu:18.04

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y emacs

# Adding wget and bzip2
RUN apt-get install -y wget bzip2

# Add sudo
RUN apt-get -y install sudo

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b
RUN rm Anaconda3-2020.02-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

# install build essentials
RUN apt-get -y install build-essential
RUN apt-get -y update

# copy RoutedFusion into image
COPY . /app
WORKDIR /app

## create anaconda environment
#RUN apt-get install -y mayavi2
RUN conda env create -f environment.yml
#
RUN echo "source activate routed-fusion" > ~/.bashrc
ENV PATH /root/anaconda3/envs/routed-fusion/bin:$PATH
#
RUN chmod +x /root/anaconda3/envs/routed-fusion/bin

# install all dependencies
RUN bash scripts/install_docker.sh
