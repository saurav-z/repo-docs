# PyTorch/XLA: Accelerate Your Deep Learning with Cloud TPUs and GPUs

**Supercharge your PyTorch models with the XLA deep learning compiler for blazing-fast performance on Cloud TPUs and GPUs!**  [Explore the PyTorch/XLA repository](https://github.com/pytorch/xla).

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that seamlessly integrates the PyTorch deep learning framework with the XLA (Accelerated Linear Algebra) deep learning compiler, enabling efficient execution on Google Cloud TPUs (Tensor Processing Units) and GPUs. This allows you to significantly speed up your model training and inference.

**Key Features:**

*   **TPU and GPU Acceleration:** Run your PyTorch models on Cloud TPUs and GPUs for faster training and inference.
*   **XLA Compiler Integration:** Leverages the XLA compiler to optimize your models for maximum performance.
*   **Easy to Use:** Simple integration into your existing PyTorch code with minimal modifications.
*   **Free Trial:** Get started with PyTorch/XLA for free on a single Cloud TPU VM using Kaggle.
*   **Comprehensive Documentation and Tutorials:** Extensive resources to help you get started and optimize your models.

## Quickstart: Get Started with PyTorch/XLA

PyTorch/XLA offers pre-built docker images and wheels, and simple code changes to enable single-process and multi-process use.

*   **Single Process:** For updating your existing training loop, make the following changes:
    ```diff
    +import torch_xla
    ...
        with torch_xla.step():
        ...
        inputs, labels = inputs.to('xla'), labels.to('xla')
        loss.backward()
        optimizer.step()
    +torch_xla.sync()
    ...
    +model.to('xla')
    ```

*   **Multi-Process:** Adapt your training loop with these modifications for multi-process training, using `torch_xla.launch`.

    ```diff
    +import torch_xla
    +import torch_xla.core.xla_model as xm
    ...
    for inputs, labels in train_loader:
        with torch_xla.step():
            inputs, labels = inputs.to('xla'), labels.to('xla')
            xm.optimizer_step(optimizer)
    ...
    +torch_xla.launch(_mp_fn, args=())
    ```

Detailed code examples can be found in the [Getting Started](#getting-started) section.

## Installation

### Installing on TPU

1.  **Create a New TPU VM:** Ensure you have a TPU VM set up.
2.  **Choose Your Python Version:**  PyTorch/XLA supports Python versions 3.8 to 3.11.
3.  **Stable Build Installation (Recommended):**

    ```bash
    # For venv
    # python3.11 -m venv py311
    # For conda
    # conda create -n py311 python=3.11
    pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
    ```
4.  **Nightly Build Installation:**
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    # Edit `cp310-cp310` to fit your desired Python version as needed
    pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
      -f https://storage.googleapis.com/libtpu-wheels/index.html
    ```

### C++11 ABI Builds

Since PyTorch/XLA 2.7 release, C++11 ABI builds are the default.  C++11 ABI wheels offer performance improvements, particularly in lazy tensor tracing.

1.  **C++11 ABI Installation for 2.6 wheels (Python 3.10 example):**

    ```bash
    pip install torch==2.6.0+cpu.cxx11.abi \
      https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
      'torch_xla[tpu]' \
      -f https://storage.googleapis.com/libtpu-releases/index.html \
      -f https://storage.googleapis.com/libtpu-wheels/index.html \
      -f https://download.pytorch.org/whl/torch
    ```
2.  **C++11 ABI Docker Image:**
    ```
    us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
    ```

## Documentation and Resources

*   **Getting Started Guides**: Guides for different modes of operation, including single process, multi-process and SPMD (Single Program, Multiple Data).
*   **Comprehensive API Guide:** [API Guide](API_GUIDE.md) for best practices.
*   **Troubleshooting Guide:**  [troubleshooting guide](docs/source/learn/troubleshoot.md) to debug and optimize your networks.
*   **User Guides:**
    *   [Documentation for the latest release](https://pytorch.org/xla)
    *   [Documentation for master branch](https://pytorch.org/xla/master)
*   **Kaggle Notebooks:** Explore example notebooks:
    *   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
    *   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)
*   **Github Doc Map:** Access various documents within the repository:
    *   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn)
    *   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators)
    *   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf)
    *   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features)
    *   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute)
    *   PJRT plugins: [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md), [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
    *   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples)

## PyTorch/XLA Tutorials

*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Reference Implementations

*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)

## Available Docker Images and Wheels

### Python Packages

PyTorch/XLA releases starting with version r2.1 will be available on PyPI. You
can now install the main build with `pip install torch_xla`. To also install the
Cloud TPU plugin corresponding to your installed `torch_xla`, install the optional `tpu` dependencies after installing the main build with

```
pip install torch_xla[tpu]
```

GPU release builds and GPU/TPU nightly builds are available in our public GCS bucket.

#### Python Packages

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

<details>

<summary>older versions</summary>

| Version | Cloud TPU VMs Wheel |
|---------|-------------------|
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

### Docker

#### TPU Docker Images

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

#### GPU Docker Images

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

To run on [compute instances with GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).

## Troubleshooting

Encountering issues?  Consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Contributing and Community

We welcome contributions! Check out the [contribution guide](CONTRIBUTING.md) to learn how to get involved.

### Contact

*   For questions directed at Meta, please send an email to opensource@fb.com.
*   For questions directed at Google, please send an email to pytorch-xla@googlegroups.com.
*   For all other questions, please open up an issue in this repository [here](https://github.com/pytorch/xla/issues).

## Additional Reads

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)