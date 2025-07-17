# PyTorch/XLA: Accelerate Your Deep Learning with Cloud TPUs and GPUs

**Supercharge your PyTorch models by harnessing the power of Google Cloud TPUs and GPUs with PyTorch/XLA!** ([Original Repository](https://github.com/pytorch/xla))

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that integrates the [PyTorch deep learning framework](https://pytorch.org/) with the [XLA deep learning compiler](https://www.tensorflow.org/xla) and [Cloud TPUs](https://cloud.google.com/tpu/), enabling you to run your models at incredible speeds. You can now also run on GPUs!

Key Features:

*   **TPU & GPU Acceleration:** Leverage the performance of Cloud TPUs and GPUs for faster training and inference.
*   **Seamless Integration:** Easily integrates with existing PyTorch code, requiring minimal changes to your training loops.
*   **XLA Compilation:** Utilizes the XLA compiler for optimized execution on target hardware.
*   **Distributed Training Support:** Supports multi-process and SPMD training for scaling your models.
*   **Community-Driven:** Actively maintained by Google, Meta, and the open-source community.

## Installation

Follow the instructions below to install PyTorch/XLA. Ensure you use a supported Python version (3.8-3.11).

### TPU Installation

**Stable Build:**

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

**Nightly Build:**

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI Builds

*As of March 18, 2024, C++11 ABI builds are default.*

*To Install a specific C++11 ABI for older releases, check below*

To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```sh
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

**Python 3.9 Wheels:**

```
https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl
```

**Python 3.10 Wheels:**

```
https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl
```

**Python 3.11 Wheels:**

```
https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl
```

To access C++11 ABI flavored docker image:

```
us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
```

### GPU Installation

```bash
pip install torch_xla[gpu] -f https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/index.html
```

## Getting Started

### Simple Single Process

1.  **Import `torch_xla`:**
    ```python
    import torch_xla
    ```
2.  **Move Model and Data to Device:**
    ```python
    model.to('xla')
    inputs, labels = inputs.to('xla'), labels.to('xla')
    ```
3.  **Wrap Training Step:**
    ```python
    with torch_xla.step():
        # Your training logic
    ```
4.  **Sync:**
    ```python
    torch_xla.sync()
    ```

### Multi-Process

1.  **Import necessary libraries:**
    ```python
    import torch_xla
    import torch_xla.core.xla_model as xm
    ```

2.  **Wrap training loop:**
    ```python
    with torch_xla.step():
        inputs, labels = inputs.to('xla'), labels.to('xla')
        xm.optimizer_step(optimizer)
    ```
3.  **Launch processes:**
    ```python
    torch_xla.launch(_mp_fn, args=())
    ```

For `DistributedDataParallel`, use the same steps for multi-process, with:

*   `dist.init_process_group("xla", init_method='xla://')`
*   `DDP(model, gradient_as_bucket_view=True)`

For more details, see the code samples in the original README, and also the guides below.

## Key Resources

*   [Documentation for the latest release](https://pytorch.org/xla)
*   [Documentation for master branch](https://pytorch.org/xla/master)
*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [GPU Guide](docs/gpu.md)

## PyTorch/XLA Tutorials

*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Reference Implementations

Explore example implementations in the [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) repository.

## Available Docker Images and Wheels

Find the latest prebuilt packages and Docker images for various configurations in the sections below:

### Python Packages

*   PyPI: `pip install torch_xla` and `pip install torch_xla[tpu]`
*   GPU wheels available at: `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/index.html`

### Docker Images

Refer to the original README for the most up-to-date versions and image tags.

## Troubleshooting

For assistance with common issues and optimization tips, see the [troubleshooting guide](docs/source/learn/troubleshoot.md).

## Contributing and Feedback

We welcome contributions! Please see the [contribution guide](CONTRIBUTING.md).  For questions, bug reports, and feature requests, please file an issue [here](https://github.com/pytorch/xla/issues).

## Additional Reads

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)