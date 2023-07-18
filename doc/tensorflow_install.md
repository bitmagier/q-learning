# Software framework stack



# Tensorflow for CUDA



## Build Tensorflow 2.12 for Nvidia GPU / CUDA

  - Ubuntu jammy
  - Python 3.10
  - bazelisk

### CUDA 11.8
   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-11-8 
   ```
### cuDNN (libcudnn8) (cuDNN runtime libraries) version 8.6 for cuda 11.8
  - It is absolutely necessary to use the exact version - otherwise we will face compile errors with tensorflow
   ```sh
   apt install libcudnn8=8.6.0.163-1+cuda11.8
   apt install libcudnn8-dev=libcudnn8=8.6.0.163-1+cuda11.8
   ```
### Tensorflow 2.12 (https://www.tensorflow.org/install)

    - Guides:
        - https://www.tensorflow.org/install/source
        - Tensorflow Rust: guide: https://github.com/tensorflow/rust/blob/master/tensorflow-sys/README.md

    - Procedure on my machine:
```sh
      git clone -b r2.12 --depth=1 https://github.com/tensorflow/tensorflow.git
      cd tensorflow
      ln -s /usr/bin/python3 ~/bin/python
      export TMP=/tmp
      ./configure  # CUDA=y, compute-capabilities=8.6, opt=-march=native
      
      # build + install python wheel package  
      bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
      # have a tee
      ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
      pip install /tmp/tensorflow_pkg/tensorflow-2.12.1-cp310-cp310-linux_x86_64.whl
    
      # Build + install shared lib
      bazel build --config=opt tensorflow:libtensorflow.so
      sudo cp -a bazel-bin/tensorflow/{libtensorflow_framework.so,libtensorflow_framework.so.2,libtensorflow_framework.so.2.12.1,libtensorflow.so,libtensorflow.so.2,libtensorflow.so.2.12.1} /usr/local/lib/
      sudo ldconfig
      tensorflow/c/generate-pc.sh --prefix=/usr/local --version=2.12.1
      sudo cp tensorflow.pc /usr/lib/pkgconfig/
      sudo chmod o+r /usr/lib/pkgconfig/tensorflow.pc
      pkg-config --libs tensorflow # checks if installed correctly
```


  - Cleanup procedure in case something goes wrong during the build:
```sh
   remove tensorflow source folder and clone again
   bazel clean --expunge
```


## NVIDIA Alternative 1: Use pre-build python wheel packages

## NVIDIA Alternative 2: Run Tensorflow NVIDIA via Docker:

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
  cd /usr/local/lib && sudo ln -s libtensorflow_framework.so.2 libtensorflow_framework.so && cd -
  sudo ldconfig
  
  tensorflow/c/generate-pc.sh --prefix=/usr/local --version=2.12.0
  sudo cp tensorflow.pc /usr/lib/pkgconfig/
  pkg-config --libs tensorflow # checks if installed correctly
```

 But f**k: unfortunately tensorflow/ROCm ignores my RX 580 with this message:
 `2023-07-16 15:44:55.968387: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2011] Ignoring visible gpu device (device: 0, name: Radeon RX 580 Series, pci bus id: 0000:0a:00.0) with AMDGPU version : gfx803. The supported AMDGPU versions are gfx1030, gfx900, gfx906, gfx908, gfx90a.`

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