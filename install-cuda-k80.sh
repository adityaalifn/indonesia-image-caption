#!/bin/bash
mkdir -p ~/dev && cd ~/dev
lspci -nnk | grep -i nvidia
sudo apt-get update
sudo apt-get -y install build-essential

wget http://us.download.nvidia.com/tesla/396.37/nvidia-diag-driver-local-repo-ubuntu1604-396.37_1.0-1_amd64.deb
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604-396.37_1.0-1_amd64.deb
sudo apt-key add /var/nvidia-diag-driver-local-repo-396.37/7fa2af80.pub

wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
mv cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb

sudo apt-get -y update
sudo apt-get -y install cuda

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
source ~/.bashrc
cd

rm -r ~/dev
