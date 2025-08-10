# PyTorch/XLA: Accelerate Your PyTorch Models with TPUs and GPUs

**Harness the power of Google Cloud TPUs and GPUs to supercharge your PyTorch deep learning projects.**  ([See the original repo](https://github.com/pytorch/xla))

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA seamlessly integrates the [PyTorch deep learning framework](https://pytorch.org/) with the [XLA deep learning compiler](https://www.tensorflow.org/xla), enabling high-performance training and inference on [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs. This powerful combination lets you train complex models faster and more efficiently.

## Key Features

*   **TPU Acceleration:** Effortlessly run your PyTorch models on Google Cloud TPUs for significantly faster training times.
*   **GPU Support:** Utilize the power of GPUs for accelerated model execution.
*   **Simplified Integration:** Minimal code changes are required to leverage XLA acceleration in your existing PyTorch code.
*   **Kaggle Integration:** Get started quickly with pre-configured Kaggle notebooks.
*   **Comprehensive Documentation:** Access detailed guides, tutorials, and API references.
*   **Active Community:** Benefit from a supportive community and actively maintained project.

## Installation

### TPU Installation

Follow these steps to install the PyTorch/XLA stable build on a new TPU VM:

**Requirements:** Builds are available for Python 3.8 to 3.13; please use one of the supported versions.

**Stable Build**

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'

# Optional: if you're using custom kernels, install pallas dependencies
pip install 'torch_xla[pallas]'
```

**Nightly Build (Python 3.11-3.13, starting with PyTorch/XLA 2.8)**

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI Builds

**Note:** Starting with PyTorch/XLA 2.7, C++11 ABI builds are the default.

For specific C++11 ABI wheel installations (e.g., Python 3.10):

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

Find wheels for other Python versions:

*   3.9:  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl
*   3.10: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl
*   3.11: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl

**C++11 ABI Docker Image:**

```
us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
```

## Getting Started

Get your models running on TPUs quickly with these code adjustments:

### Simple Single Process

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

**For DistributedDataParallel, make the following changes:**

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

## Useful Resources

*   **Quickstart:** [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   **Documentation:** [Latest Release Documentation](https://pytorch.org/xla) & [Master Branch Documentation](https://pytorch.org/xla/master)
*   **Tutorials:**
    *   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
    *   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
    *   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    *   [GPU guide](docs/gpu.md)
*   **Reference Implementations:** [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)
*   **Troubleshooting:** [Troubleshooting Guide](docs/source/learn/troubleshoot.md)
*   **Additional Reads:** Explore performance debugging and lazy tensor introductions in the Additional Reads section of the original README.

## Available Docker Images and Wheels

**See the original README for a comprehensive list of available Docker images and Python packages.**  You can find these under the "Available docker images and wheels" section.

## Github Doc Map

Our github contains many useful docs on working with different aspects of PyTorch XLA, here is a list of useful docs spread around our repository:

- [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): docs for learning concepts associated with XLA, troubleshooting, pjrt, eager mode, and dynamic shape.
- [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): references to `GPU` and `TPU` accelerator documents.
- [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): documentation about performance specific aspects of PyTorch/XLA such as: `AMP`, `DDP`, `Dynamo`, Fori loop, `FSDP`, quantization, recompilation, and `SPMD`
- [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): documentation on distributed torch, pallas, scan, stable hlo, and triton.
- [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): documents on setting up PyTorch for development, and guides for lowering operations.
- PJRT plugins:
  - [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
  - [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
- [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
  - [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples


## Contributing and Feedback

The PyTorch/XLA team welcomes contributions!  See the [contribution guide](CONTRIBUTING.md) for details.  Report issues, ask questions, and provide feedback on the project's [GitHub issues page](https://github.com/pytorch/xla/issues).

## Disclaimer

This project is jointly maintained by Google, Meta, and individual contributors.