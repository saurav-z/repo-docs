# PyTorch/XLA: Accelerate Your PyTorch Models with Cloud TPUs and GPUs

**Supercharge your deep learning workflows by seamlessly integrating PyTorch with XLA for blazing-fast performance on Cloud TPUs and GPUs.**  Explore the [PyTorch/XLA GitHub Repository](https://github.com/pytorch/xla) for detailed information and resources.

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA empowers you to harness the power of Google's [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs directly within the familiar PyTorch framework.  It utilizes the [XLA deep learning compiler](https://www.tensorflow.org/xla) to optimize your models for enhanced speed and efficiency.

## Key Features

*   **TPU Acceleration:** Run your PyTorch models on Cloud TPUs for significant performance gains.
*   **GPU Support:** Utilize GPUs for accelerated training and inference.
*   **Seamless Integration:**  Easy integration with PyTorch code, minimizing the need for extensive code modifications.
*   **Optimized Performance:** Leverage the XLA compiler for efficient execution and resource utilization.
*   **Cloud TPU VM Compatibility:** Try it out now, for free, on a single Cloud TPU VM with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

## Installation

### TPU Installation

Install the stable or nightly builds of PyTorch/XLA in a new TPU VM, ensuring you select a supported Python version (3.8 to 3.13).

**Stable Build (Python 3.8 - 3.11):**

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'

# Optional: if you're using custom kernels, install pallas dependencies
pip install 'torch_xla[pallas]'
```

**Nightly Build (Python 3.11 - 3.13):**
*As of 07/16/2025 and starting from Pytorch/XLA 2.8 release, PyTorch/XLA will provide nightly and release wheels for Python 3.11 to 3.13*

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI Builds (Recommended)

*As of 03/18/2025 and starting from Pytorch/XLA 2.7 release, C++11 ABI builds are the default and we no longer provide wheels built with pre-C++11 ABI.*

For the best performance, especially with lazy tensor tracing, use the C++11 ABI builds.

**C++11 ABI (Python 3.10 Example):**

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

Python 3.9 and 3.11 C++11 ABI wheels are also available:

*   3.9: `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl`
*   3.10: `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl`
*   3.11: `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl`

**C++11 ABI Docker Image:**

```bash
docker pull us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
```

## Getting Started

Follow these guides to integrate PyTorch/XLA into your existing training loops:

*   **Single Process:**  Train on a single GPU/TPU with one Python interpreter.
*   **Multi-Process:**  Utilize multiple Python interpreters for distributed training across multiple GPUs/TPUs (not compatible with SPMD).

### Simple Single Process Example

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

### Multi-Processing Example

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
            xm.optimizer_step(optimizer)

if __name__ == '__main__':
    torch_xla.launch(_mp_fn, args=())
```

For `DistributedDataParallel`, make the following adjustments:

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

## Documentation and Resources

*   **API Guide:**  [API_GUIDE.md](API_GUIDE.md) - best practices for XLA device programming.
*   **Comprehensive User Guides:**
    *   [Documentation for the latest release](https://pytorch.org/xla)
    *   [Documentation for master branch](https://pytorch.org/xla/master)
*   **Tutorials:**
    *   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
    *   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
    *   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    *   [GPU guide](docs/gpu.md)
*   **Reference Implementations:** [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) - examples for training LLMs and diffusion models.
*   **Additional Reads:**
    *   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
    *   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
    *   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
    *   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Available Docker Images and Wheels

Find the latest versions and installation instructions for PyPI packages, GPU wheels, and Docker images [here](https://github.com/pytorch/xla#available-docker-images-and-wheels).

## Troubleshooting

If you encounter performance issues, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging tips and optimization strategies.

## Contributing

We welcome contributions!  Refer to the [contribution guide](CONTRIBUTING.md) to get started.

## Community and Support

*   **Feedback:** File issues on [GitHub](https://github.com/pytorch/xla/issues) for questions, bug reports, and feature requests.
*   **Contact:**
    *   For Meta-related questions: opensource@fb.com.
    *   For Google-related questions: pytorch-xla@googlegroups.com.

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)