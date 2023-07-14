# Software framework stack

__Required installs:__

# Tensorflow 2.12 GPU

- Python

- CUDA 11.8
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-11-8
   ```
- cuDNN (libcudnn8) (cuDNN runtime libraries) version 8.6 for cuda 11.8
    - https://developer.nvidia.com/rdp/cudnn-download
   ```
   apt install ./cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb`
   apt install /var/cudnn-local-repo-ubuntu2204-8.6.0.163/libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb
   apt install /var/cudnn-local-repo-ubuntu2204-8.6.0.163/libcudnn8-dev_8.6.0.163-1+cuda11.8_amd64.deb
   ```
- Install / Compile Tensorflow 2.12 (https://www.tensorflow.org/install)

# Alternative: Tensorflow via Docker:

- [docker setup](https://docs.docker.com/desktop/install/linux-install/)
- [nvidia-docker setup](https://github.com/NVIDIA/nvidia-docker)
- `docker run --gpus all -it --rm tensorflow/tensorflow:2.12.0-gpu bash`
 
