# PyTorch/XLA: Accelerate Your PyTorch Models with TPUs and GPUs

**Supercharge your deep learning workflows by seamlessly integrating PyTorch with Google Cloud TPUs and GPUs using PyTorch/XLA!**  Visit the [PyTorch/XLA GitHub repository](https://github.com/pytorch/xla) for the source code and more information.

Key Features:

*   **TPU Acceleration:** Leverage the power of Cloud TPUs for significantly faster training and inference of your PyTorch models.
*   **GPU Support:** Extend the reach to run PyTorch on a wide range of GPUs.
*   **Easy Integration:** Integrate with minimal code changes, making it simple to adopt XLA.
*   **Kaggle Ready:** Experiment and get started easily on Kaggle with pre-built notebooks.
*   **Comprehensive Documentation:** Access detailed guides and tutorials for setup, optimization, and troubleshooting.
*   **Active Community:** Engage with the PyTorch/XLA community through GitHub issues and forums.

## Installation

### TPU

To install PyTorch/XLA for Cloud TPUs, follow these steps:

**Prerequisites:** Ensure you're using a supported Python version (3.8 to 3.13, see notes below).

**Stable Builds:**

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'

# Optional: If using custom kernels, install Pallas dependencies:
pip install 'torch_xla[pallas]'
```

**Nightly Builds:**

*   **Important Note:** As of 07/16/2025 and starting from Pytorch/XLA 2.8 release, PyTorch/XLA will provide nightly and release wheels for Python 3.11 to 3.13.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI Builds

*   **Important Note:** As of 03/18/2025 and starting from Pytorch/XLA 2.7 release, C++11 ABI builds are the default and wheels with pre-C++11 ABI are no longer provided.

  To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

  The above command works for Python 3.10. We additionally have Python 3.9 and 3.11
wheels:

*   3.9: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl
*   3.10: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl
*   3.11: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl

  To access C++11 ABI flavored docker image:

```
us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
```

## GitHub Documentation Map

The PyTorch/XLA repository houses a wealth of documentation. Here are key resource areas:

*   [Learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): Concepts, troubleshooting, PJRT, eager mode, and dynamic shapes.
*   [Accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): GPU and TPU specific documents.
*   [Performance](https://github.com/pytorch/xla/tree/master/docs/source/perf): AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   [Features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed training, Pallas, scan, stable HLO, and Triton.
*   [Contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Development setup and operation lowering guides.
*   PJRT Plugins:
    *   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    *   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documentation
    *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

## Getting Started

PyTorch/XLA supports both single-process and multi-process training:

*   **Single Process:** One Python interpreter controls a single GPU/TPU.
*   **Multi-Process:** Multiple Python interpreters are launched, each managing a GPU/TPU.  See SPMD guide for more on SPMD.

### Single Process Example

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

### Multi-Process Example

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

### Multi-Process with DistributedDataParallel

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

## Tutorials and Guides

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU Guide](docs/gpu.md)
*   [API Guide](API_GUIDE.md)

## Reference Implementations

*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes): Examples for LLM and diffusion model training and serving.

## Available Docker Images and Wheels

Find the latest pre-built packages and images:

### Python Packages

PyTorch/XLA packages are available on PyPI and through GCS buckets.  Install the core package:
```bash
pip install torch_xla
pip install torch_xla[tpu]
```
See the README for specific installation details.

### Docker

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```
And more images are available in the above sections.

## Troubleshooting

Refer to the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and performance optimization tips.

## Feedback and Contribution

*   **Feedback:**  Please file issues on GitHub for questions, bug reports, and feature requests.
*   **Contribution:** See the [contribution guide](CONTRIBUTING.md) to get involved.

## Disclaimer

This project is jointly maintained by Google, Meta, and individual contributors.

## Additional Resources

*   [Performance Debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy Tensor Intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling Deep Learning Workloads with PyTorch/XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch Models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)