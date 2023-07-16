# Software framework stack

# Tensorflow for Nvidia GPU / CUDA

  - Ubuntu jammy
  - Python 3.10

## CUDA 11.8
   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-11-8
   ```
## cuDNN (libcudnn8) (cuDNN runtime libraries) version 8.6 for cuda 11.8
    - https://developer.nvidia.com/rdp/cudnn-download
   ```sh
   apt install ./cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb`
   apt install /var/cudnn-local-repo-ubuntu2204-8.6.0.163/libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb
   apt install /var/cudnn-local-repo-ubuntu2204-8.6.0.163/libcudnn8-dev_8.6.0.163-1+cuda11.8_amd64.deb
   ```
## Tensorflow 2.12 (https://www.tensorflow.org/install)

  - Compile Tensorflow 2.12.1 Guide guide: https://github.com/tensorflow/rust/blob/master/tensorflow-sys/README.md
  
  - On my machine with CUDA: 
      ```sh
      git clone https://github.com/tensorflow/tensorflow.git
      cd tensorflow
      git checkout v2.12.1
      ./configure  # CUDA = y
      bazelisk build --compilation_mode=opt --copt=-march=native  tensorflow:libtensorflow.so
      # have a tee
      sudo cp -a bazel-bin/tensorflow/{libtensorflow_framework.so,libtensorflow_framework.so.2,libtensorflow_framework.so.2.12.1,libtensorflow.so,libtensorflow.so.2,libtensorflow.so.2.12.1} /usr/local/lib/
      sudo ldconfig
      tensorflow/c/generate-pc.sh --prefix=/usr/local --version=2.12.1
      sudo cp tensorflow.pc /usr/lib/pkgconfig/
      pkg-config --libs tensorflow # checks if installed correctly
      ```

# NVIDIA Alternative: Run Tensorflow NVIDIA via Docker:

- [docker setup](https://docs.docker.com/desktop/install/linux-install/)
- [nvidia-docker setup](https://github.com/NVIDIA/nvidia-docker)
- `docker run --gpus all -it --rm tensorflow/tensorflow:2.12.0-gpu bash`


# Tensorflow + AMD GPU / ROCm

  ## Install AMD driver + ROCm 5.6
  https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/quick-start.md
  ```
  wget https://repo.radeon.com/amdgpu-install/5.6/ubuntu/jammy/amdgpu-install_5.6.50600-1_all.deb
  sudo apt-get install ./amdgpu-install_5.6.50600-1_all.deb
  sudo amdgpu-install --usecase=rocm
  ```

  ## Build Tensorflow 2.12 for ROCm
  ```sh
  git clone git@github.com:ROCmSoftwarePlatform/tensorflow-upstream.git
  cd tensorflow-upstream
  git checkout r2.12-rocm-enhanced  
  ./build_rocm_python3 # + CTRL-C : we just need the created config files  
  bazelisk build --compilation_mode=opt --copt=-march=native tensorflow:libtensorflow.so
  # have a tee
  sudo cp -a bazel-bin/tensorflow/{libtensorflow_framework.so.2,libtensorflow_framework.so.2.12.0,libtensorflow.so,libtensorflow.so.2,libtensorflow.so.2.12.0} /usr/local/lib/
  sudo ldconfig
  
  tensorflow/c/generate-pc.sh --prefix=/usr/local --version=2.12.0
  sudo cp tensorflow.pc /usr/lib/pkgconfig/
  pkg-config --libs tensorflow # checks if installed correctly
  ```

# Rust
- follow https://www.rust-lang.org/tools/install
- `sudo apt install libssl-dev libfontconfig-dev`

# Run integration test
```sh
git clone git@github.com:bitmagier/q-learning-breakout.git
cd q-learning-breakout
cd tf_model
python3 create_ql_model_ballgame_5x5x3_4_32.py
cd ..

cargo test --test learn_ballgame --release
```