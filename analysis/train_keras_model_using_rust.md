
# Software framework stack
[Article: training-keras-models-using-the-rust-tensorflow-bindings](https://towardsdatascience.com/training-keras-models-using-the-rust-tensorflow-bindings-941791249a7)

__Required installs:__

- Tensorflow
  - native:
    - [tensorflow](https://www.tensorflow.org/install/docker)
    - nvidia-cuda-toolkit
    - libcudnn8 (cuDNN runtime libraries)

  - or via docker:
    - [docker setup](https://docs.docker.com/desktop/install/linux-install/)
    - [nvidia-docker setup](https://github.com/NVIDIA/nvidia-docker)
    - `docker run --gpus all -it --rm tensorflow/tensorflow:2.12.0-gpu bash`
 
