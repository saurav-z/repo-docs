# PyTorch/XLA: Accelerate Your PyTorch Models with XLA on TPUs and GPUs

**Harness the power of XLA, the deep learning compiler, to supercharge your PyTorch models and run them efficiently on Cloud TPUs and GPUs.** (🔗 [Original Repo](https://github.com/pytorch/xla))

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

## Key Features

*   **TPU Acceleration:** Seamlessly integrate PyTorch with Cloud TPUs for blazing-fast training and inference.
*   **GPU Support:** Utilize the XLA compiler to optimize performance on compatible GPUs.
*   **Simplified Integration:** Easily adapt your existing PyTorch code with minimal changes.
*   **Optimized Performance:** Benefit from XLA's advanced compilation and optimization techniques.
*   **Open Source:** Join a vibrant community and contribute to the development of PyTorch/XLA.

## Getting Started

### Installation

**For Cloud TPUs:**  Ensure you use a supported Python version (3.8-3.13).

**Stable Build Installation:**

```bash
pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
# Optional: Install pallas dependencies if you're using custom kernels
pip install 'torch_xla[pallas]'
```

**Nightly Build Installation:**

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

**For C++11 ABI builds**, refer to the instructions in the original README for Python 3.9-3.11.  C++11 ABI builds and docker images can improve performance.

### Quick Start Guides

*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

### Code Examples

**Single Process:**

```python
import torch_xla

# ... your training loop ...
for inputs, labels in train_loader:
    with torch_xla.step():
        inputs, labels = inputs.to('xla'), labels.to('xla')
        # ... rest of your training code ...
# ... After the training loop
torch_xla.sync()
# Move the model paramters to your XLA device
model.to('xla')
```

**Multi-Process:**

```python
import torch_xla
import torch_xla.core.xla_model as xm

def _mp_fn(index):
    # Move the model paramters to your XLA device
    model.to('xla')
    for inputs, labels in train_loader:
        with torch_xla.step():
            inputs, labels = inputs.to('xla'), labels.to('xla')
            # ... training code ...
            xm.optimizer_step(optimizer)

# Launch training
torch_xla.launch(_mp_fn, args=())
```

## Documentation & Resources

*   **[Comprehensive User Guides](https://pytorch.org/xla)** (Documentation for the latest release)
*   **[Master Branch Documentation](https://pytorch.org/xla/master)**
*   **[Github Doc Map](#github-doc-map)**

### GitHub Doc Map

*   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn):  Learning XLA concepts.
*   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators):  GPU and TPU accelerator docs.
*   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf):  Performance optimizations (AMP, DDP, etc.).
*   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features):  Distributed torch, pallas, etc.
*   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute):  Development setup and guides.
*   PJRT plugins: [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md), [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): Torchax documentation
    *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): Torchax examples

## Available Docker Images and Wheels

Refer to the original README for detailed information on available Python packages, docker images and wheels, including nightly and release builds for TPUs and GPUs, organized by version.

## Troubleshooting

Experiencing issues? Consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Contribute & Get Involved

We welcome contributions! See the [contribution guide](CONTRIBUTING.md) for details.

## Feedback

Share your questions, bug reports, and feature requests by [filing an issue](https://github.com/pytorch/xla/issues).

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)