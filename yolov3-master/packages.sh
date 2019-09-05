#!/bin/sh
apt-get update 
apt-get upgrade -y
apt-get install 'python3.6' -y
apt-get install 'python3-pip' -y
apt-get install 'libsm6' -y
apt-get install 'libxrender1' -y
pip3 install cython
pip3 install numpy
pip3 install torch
pip3 install torchvision
pip3 install opencv-contrib-python
pip3 install azure-cognitiveservices-vision-customvision
pip3 install azure
pip3 install numpy
pip3 install opencv-python
pip3 install matplotlib
pip3 install pycocotools
pip3 install tqdm
pip3 install tb-nightly
pip3 install future
pip3 install Pillow
