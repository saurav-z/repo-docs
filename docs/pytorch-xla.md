# PyTorch/XLA: Accelerate Your Deep Learning with TPUs and GPUs

**Supercharge your PyTorch models by seamlessly leveraging the power of Google Cloud TPUs and GPUs using the XLA compiler.** [(View on GitHub)](https://github.com/pytorch/xla)

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that enables you to run PyTorch deep learning models on [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs, utilizing the [XLA (Accelerated Linear Algebra) deep learning compiler](https://www.tensorflow.org/xla) for optimized performance. This powerful combination offers significant speedups for training and inference, especially for large and complex models.  You can even try it for free on a single Cloud TPU VM with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

## Key Features

*   **TPU Acceleration:** Train your PyTorch models on Google Cloud TPUs for significantly faster training times.
*   **GPU Support:** Run your models on GPUs leveraging the XLA compiler, providing optimized performance.
*   **Ease of Use:** Simple integration with existing PyTorch code, minimizing the need for extensive code modifications.
*   **XLA Compiler:** Benefit from the performance optimizations provided by the XLA deep learning compiler.
*   **Free Trial:** Get started quickly with free access on Kaggle.
*   **Comprehensive Documentation:** Access a wealth of resources including tutorials, guides and examples.

## Getting Started

### Installation

Install the stable or nightly builds of PyTorch/XLA. Builds are available for Python 3.8 to 3.11.

#### Stable Build (TPU)
```bash
pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

#### Nightly Build (TPU)
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

#### Stable Build (GPU)
See the instructions in the [Available docker images and wheels](#available-docker-images-and-wheels) section to identify the relevant wheel for your CUDA and Python version.

### Code Integration

Adapt your existing PyTorch training loop with a few simple modifications.

**Single Process Example:**

```python
import torch_xla

def train(model, training_data, ...):
  ...
  for inputs, labels in train_loader:
    with torch_xla.step():
      inputs, labels = training_data[i]
      inputs, labels = inputs.to('xla'), labels.to('xla')
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

  torch_xla.sync()
  ...

if __name__ == '__main__':
  ...
  # Move the model paramters to your XLA device
  model.to('xla')
  train(model, training_data, ...)
  ...
```

**Multi-Process Example:**

```python
import torch_xla
import torch_xla.core.xla_model as xm

def _mp_fn(index):
  # Move the model paramters to your XLA device
  model.to('xla')

  for inputs, labels in train_loader:
    with torch_xla.step():
      # Transfer data to the XLA device. This happens asynchronously.
      inputs, labels = inputs.to('xla'), labels.to('xla')
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      # `xm.optimizer_step` combines gradients across replicas
      xm.optimizer_step(optimizer)

if __name__ == '__main__':
  # torch_xla.launch automatically selects the correct world size
  torch_xla.launch(_mp_fn, args=())
```

Detailed guides are available in the [Getting Started](#getting-started) section, as well as in the documentation.

## Available Docker Images and Wheels

### Python Packages

PyTorch/XLA releases are available on PyPI, and nightly builds are available from our public GCS bucket. Find the appropriate versions for your hardware (TPU or GPU) and your Python and CUDA versions.

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp39-cp39-linux_x86_64.whl` |
| nightly (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |

#### Use nightly build
You can also add `yyyymmdd` like `torch_xla-2.8.0.devyyyymmdd` (or the latest dev version)
to get the nightly wheel of a specified date. Here is an example:
```
pip3 install torch==2.8.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev20250423-cp310-cp310-linux_x86_64.whl
```

The torch wheel version `2.8.0.dev20250423+cpu` can be found at https://download.pytorch.org/whl/nightly/torch/.

### Docker
NOTE: Since PyTorch/XLA 2.7, all builds will use the C++11 ABI by default
| Version | Cloud TPU VMs Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_tpuvm` |
| 2.6 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm` |
| 2.6 (C++11 ABI) | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11` |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_tpuvm` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm` |
| 2.3 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_tpuvm` |
| 2.2 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_tpuvm` |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm` |
| nightly python | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm` |

To use the above dockers, please pass `--privileged --net host --shm-size=16G` along. Here is an example:
```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```
<br/>

| Version | GPU CUDA 12.6 Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` |


<br/>


| Version | GPU CUDA 12.4 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.4` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.4` |

<br/>


| Version | GPU CUDA 12.1 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.1` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.1` |
| 2.3 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_cuda_12.1` |
| 2.2 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1` |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_12.1` |
| nightly | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1` |
| nightly at date | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1_YYYYMMDD` |

<br/>

| Version | GPU CUDA 11.8 + Docker |
| --- | ----------- |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_11.8` |
| 2.0 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.0_3.8_cuda_11.8` |

<br/>

## Documentation and Tutorials

*   [Documentation for the latest release](https://pytorch.org/xla)
*   [Documentation for master branch](https://pytorch.org/xla/master)
*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Troubleshooting

Refer to the [troubleshooting guide](docs/source/learn/troubleshoot.md) for solutions to common issues.

## Contribution and Feedback

The PyTorch/XLA team welcomes contributions! Please refer to the [contribution guide](CONTRIBUTING.md).
For issues, bug reports, and feature requests, please open an issue [here](https://github.com/pytorch/xla/issues).

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)