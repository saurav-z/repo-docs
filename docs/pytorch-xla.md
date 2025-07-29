# PyTorch/XLA: Accelerate Your PyTorch Models with Cloud TPUs and GPUs

**Supercharge your deep learning workflows by seamlessly integrating PyTorch with Google Cloud TPUs and GPUs using PyTorch/XLA!** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/)  [View the original repository](https://github.com/pytorch/xla)

PyTorch/XLA (XLA stands for **X**LA **L**inear **A**lgebra) is a Python package that connects the [PyTorch](https://pytorch.org/) deep learning framework with the [XLA deep learning compiler](https://www.tensorflow.org/xla) and [Cloud TPUs](https://cloud.google.com/tpu/). This enables you to accelerate your PyTorch models and run them efficiently on Google Cloud TPUs and GPUs. It is a flexible framework that allows you to run your model on various accelerators such as CPUs, GPUs and TPUs.

## Key Features

*   **TPU Acceleration:** Harness the power of Cloud TPUs for significantly faster training and inference.
*   **GPU Support:** Leverage the power of your existing GPUs for increased performance.
*   **Simplified Integration:** Easy to integrate into your existing PyTorch code with minimal changes.
*   **Performance Optimization:** Achieve significant speedups through XLA compilation and optimization techniques.
*   **Flexible Deployment:** Run your models on various platforms and architectures.

## Installation

### Prerequisites

*   Python 3.8 to 3.13 (ensure you're using a supported version)
*   Ensure you have a new TPU VM or a supported GPU environment.

### TPU Installation

1.  **Create a Python Environment (Recommended):**

    ```bash
    # - for venv
    # python3.11 -m venv py311
    # - for conda
    # conda create -n py311 python=3.11
    ```

2.  **Install PyTorch/XLA:**

    *   **Stable Release (Recommended):**

        ```bash
        pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
        ```

    *   **Nightly Release (For latest features and bug fixes, may be less stable):**

        ```bash
        pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
        # Edit `cp310-cp310` to fit your desired Python version as needed
        pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
          -f https://storage.googleapis.com/libtpu-wheels/index.html
        ```

### C++11 ABI Builds (Performance Optimization)
Use the C++11 wheels for improved performance, especially if your model is CPU-bound during tracing.

To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```sh
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

### GPU Installation

Refer to the available GPU wheels in the table in the "Available docker images and wheels" section.

*   **CUDA:**  Ensure you have the correct CUDA drivers installed.  Choose the appropriate wheel for your CUDA version (e.g., CUDA 12.6), Python version, and GPU platform.

## Getting Started

### Single Process

To adapt your existing training loop:

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
  model.to('xla')
  train(model, training_data, ...)
  ...
```

### Multi-Process

Adapt your existing training loop:

```python
import torch_xla
import torch_xla.core.xla_model as xm

def _mp_fn(index):
  ...

  model.to('xla')

  for inputs, labels in train_loader:
    with torch_xla.step():
      inputs, labels = inputs.to('xla'), labels.to('xla')
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      xm.optimizer_step(optimizer) # Use xm.optimizer_step for XLA

if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=()) # Use torch_xla.launch
```

For `DistributedDataParallel`:

```python
import torch.distributed as dist
import torch_xla
import torch_xla.distributed.xla_backend

def _mp_fn(rank):
  ...

  dist.init_process_group("xla", init_method='xla://')

  model.to('xla')
  ddp_model = DDP(model, gradient_as_bucket_view=True)

  for inputs, labels in train_loader:
    with torch_xla.step():
      inputs, labels = inputs.to('xla'), labels.to('xla')
      optimizer.zero_grad()
      outputs = ddp_model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
```

### Explore the documentation

*   [API Guide](API_GUIDE.md) for best practices for XLA devices.
*   [Documentation for the latest release](https://pytorch.org/xla)
*   [Documentation for master branch](https://pytorch.org/xla/master)

## Tutorials & Resources

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Available Docker Images and Wheels

### Python Packages

PyTorch/XLA wheels are available on PyPI. For the Cloud TPU plugin:
```bash
pip install torch_xla[tpu]
```

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (Python 3.12) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
| nightly (Python 3.13) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |

#### Use nightly build

You can also add `yyyymmdd` like `torch_xla-2.9.0.devyyyymmdd` (or the latest dev version)
to get the nightly wheel of a specified date. Here is an example:

```
pip3 install torch==2.9.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev20250423-cp310-cp310-linux_x86_64.whl
```

The torch wheel version `2.9.0.dev20250423+cpu` can be found at https://download.pytorch.org/whl/nightly/torch/.

<details>
<summary>older versions</summary>
| Version | Cloud TPU VMs Wheel |
|---------|-------------------|
| 2.7 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.6 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.4 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.3 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.3.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.2 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (XRT + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/xrt/tpuvm/torch_xla-2.1.0%2Bxrt-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.1.0-cp38-cp38-linux_x86_64.whl` |

<br/>

| Version | GPU Wheel |
| --- | ----------- |
| 2.5 (CUDA 12.1 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.1 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.4 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.4/torch_xla-2.5.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.4 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.4/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.4 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.4/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.4 (CUDA 12.1 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.4 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.4 (CUDA 12.1 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.3 (CUDA 12.1 + Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.3.0-cp38-cp38-manylinux_2_28_x86_64.whl` |
| 2.3 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.3.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.3 (CUDA 12.1 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.3.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.2 (CUDA 12.1 + Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.2.0-cp38-cp38-manylinux_2_28_x86_64.whl` |
| 2.2 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 + CUDA 11.8 | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/11.8/torch_xla-2.1.0-cp38-cp38-manylinux_2_28_x86_64.whl` |
| nightly + CUDA 12.0 >= 2023/06/27| `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.0/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |

</details>

### Docker Images

NOTE: Since PyTorch/XLA 2.7, all builds will use the C++11 ABI by default

To run the docker, use the command: `docker run --privileged --net host --shm-size=16G`

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

| Version | GPU CUDA 12.6 Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` |

| Version | GPU CUDA 12.4 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.4` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.4` |

| Version | GPU CUDA 12.1 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.1` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.1` |
| 2.3 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_cuda_12.1` |
| 2.2 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1` |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_12.1` |
| nightly | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1` |
| nightly at date | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1_YYYYMMDD` |

| Version | GPU CUDA 11.8 + Docker |
| --- | ----------- |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_11.8` |
| 2.0 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.0_3.8_cuda_11.8` |

## Troubleshooting

If you encounter issues, the [troubleshooting guide](docs/source/learn/troubleshoot.md) provides valuable suggestions for debugging and optimizing your networks.

## Feedback & Contributions

The PyTorch/XLA team welcomes your feedback!  Please file issues on [GitHub](https://github.com/pytorch/xla/issues) for questions, bug reports, and feature requests. Contribute to the project following the [contribution guide](CONTRIBUTING.md).

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)