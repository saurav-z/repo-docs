# PyTorch/XLA: Accelerate Your PyTorch Models with TPUs and GPUs

[PyTorch/XLA](https://github.com/pytorch/xla) empowers you to seamlessly run your PyTorch models on Google Cloud TPUs and GPUs, significantly accelerating your deep learning workflows.

**Current CI status:**  [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

## Key Features

*   **TPU and GPU Acceleration:** Run your PyTorch models on Cloud TPUs for cutting-edge performance or GPUs for flexible acceleration.
*   **Easy Integration:** Integrate PyTorch/XLA with minimal code changes to your existing PyTorch training loops.
*   **Simplified Installation:**  Easy-to-use installation instructions for various environments, including TPU VMs and GPU instances.
*   **Comprehensive Documentation:** Extensive documentation, tutorials, and examples to help you get started and optimize your models.
*   **Active Community:** Benefit from a vibrant community and dedicated support for questions and contributions.

## Installation

### TPU

**Prerequisites:** Builds are available for Python 3.8 to 3.13; ensure you use a supported Python version.

**Stable Builds (Recommended):**

```bash
pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'

# Optional: Install pallas dependencies if you're using custom kernels
pip install 'torch_xla[pallas]'
```

**Nightly Builds (For bleeding-edge features and latest updates):**

**Note**: Starting with the 2.8 release, nightly and release wheels are available for Python 3.11 to 3.13.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Adjust `cp310-cp310` to match your Python version
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI Builds

**Default ABI**: Starting from PyTorch/XLA 2.7, C++11 ABI builds are the default.  C++11 ABI builds can improve lazy tensor tracing performance.

**To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):**

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

**Available C++11 ABI wheels for other Python versions**:

*   3.9: `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl`
*   3.10: `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl`
*   3.11: `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl`

**C++11 ABI Docker Image:**

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11 /bin/bash
```

## GitHub Doc Map

Find comprehensive documentation, guides, and examples within the repository:

*   **[docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn):** Concepts, troubleshooting, PJRT, eager mode, and dynamic shape.
*   **[docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators):** GPU and TPU accelerator references.
*   **[docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf):** Performance aspects: AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   **[docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features):** Distributed torch, Pallas, scan, stable HLO, and Triton documentation.
*   **[docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute):** Setting up PyTorch for development and guides for lowering operations.
*   **PJRT Plugins:** CPU ([CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)), and CUDA ([CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)) documentation.
*   **[torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs):** torchax documents and examples.

## Getting Started

Choose from single-process or multi-process training setups, or explore SPMD (Single Program, Multiple Data).

### Simple Single Process

Modify your training loop as follows:

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

### Multi-Processing

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

### DistributedDataParallel (DDP)

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

## Tutorials

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU Guide](docs/gpu.md)

## Reference Implementations

Explore LLM and diffusion model examples in the [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) repository.

## Available Docker Images and Wheels

### Python Packages

Install PyTorch/XLA using `pip install torch_xla`, and include the Cloud TPU plugin with `pip install torch_xla[tpu]`. Find pre-built wheels for various configurations:

**TPU Wheels**

| Version | Cloud TPU VMs Wheel |
| --- | ----------- |
| 2.7 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.6 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.4 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.3 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.3.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.2 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (XRT + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/xrt/tpuvm/torch_xla-2.1.0%2Bxrt-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.1.0-cp38-cp38-linux_x86_64.whl` |

**GPU Wheels**

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (Python 3.12) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
| nightly (Python 3.13) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |

**Use Nightly Build**

```bash
pip3 install torch==2.9.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev20250423-cp310-cp310-linux_x86_64.whl
```

### Docker Images

**TPU Docker Images**

Use the Docker images for pre-configured environments (remember to pass `--privileged --net host --shm-size=16G` when running the docker container).

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

**GPU Docker Images**

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

If you encounter performance issues, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Providing Feedback

We welcome your feedback!  Please report issues, ask questions, or suggest features via the [GitHub issue tracker](https://github.com/pytorch/xla/issues).

## Contributing

See the [contribution guide](CONTRIBUTING.md) for guidelines on contributing.

## Disclaimer

This project is maintained by Google, Meta, and community contributors.

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy Tensor Intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)