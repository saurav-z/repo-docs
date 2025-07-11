# PyTorch/XLA: Accelerate Your Deep Learning with Cloud TPUs and GPUs

**Supercharge your PyTorch models with PyTorch/XLA, enabling blazing-fast training and inference on Cloud TPUs and GPUs.**  [Visit the PyTorch/XLA repository on GitHub](https://github.com/pytorch/xla) for the latest updates and to contribute!

**Key Features:**

*   **Seamless Integration:** Connects the PyTorch framework with Google Cloud TPUs and GPUs using the XLA (Accelerated Linear Algebra) compiler.
*   **High Performance:** Enables significant speedups for training and inference, leveraging the power of specialized hardware.
*   **Flexible Deployment:** Supports various deployment options, including Cloud TPU VMs and GPUs, allowing you to choose the best infrastructure for your needs.
*   **Easy to Use:** Provides straightforward installation and integration with existing PyTorch code, minimizing the learning curve.
*   **Comprehensive Documentation:** Offers extensive guides and tutorials to help you get started and optimize your models.
*   **Community Support:** Benefits from an active community and dedicated team, ensuring you have the resources you need.

## Getting Started

PyTorch/XLA supports training models on both single and multi-process environments.
For more in-depth information, please consult the [SPMD guide](https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_basic.md).

### Quickstart Examples

Adapt your existing training loop by following these simple changes to quickly start utilizing PyTorch/XLA:

**Single Process**
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

**Multi-Process**
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

## Installation

Install the stable or nightly versions by following the steps provided.

### TPU Installation

1.  **Install Stable Build:**

    ```bash
    # For venv
    # python3.11 -m venv py311
    # For conda
    # conda create -n py311 python=3.11
    pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
    ```

2.  **Install Nightly Build:**

    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    # Edit `cp310-cp310` to fit your desired Python version as needed
    pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
    -f https://storage.googleapis.com/libtpu-wheels/index.html
    ```

### C++11 ABI Builds

Starting from PyTorch/XLA 2.7, C++11 ABI builds are the default.

To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```
**Python 3.9:**
```bash
https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl
```

**Python 3.10:**
```bash
https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl
```

**Python 3.11:**
```bash
https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl
```

To access C++11 ABI flavored docker image:

```bash
us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
```

## Useful Links

*   [PyTorch/XLA Documentation](https://pytorch.org/xla)
*   [Kaggle Notebooks](https://github.com/pytorch/xla/tree/master/contrib/kaggle)
*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)
*   [Troubleshooting guide](docs/source/learn/troubleshoot.md)
*   [Contribution guide](CONTRIBUTING.md)

## Documentation Map

Explore the PyTorch/XLA repository by visiting these helpful documentation pages:

*   [Learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): XLA concepts, troubleshooting, PJRT, eager mode, and dynamic shape.
*   [Accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): Information about GPU and TPU accelerators.
*   [Performance](https://github.com/pytorch/xla/tree/master/docs/source/perf): Performance-specific aspects such as AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   [Features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed torch, pallas, scan, stable hlo, and triton.
*   [Contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Development setup, and guides for lowering operations.
*   [PJRT plugins](https://github.com/pytorch/xla/blob/master/plugins): CPU and CUDA PJRT plugins.
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): Torchax documentation
*   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): Torchax examples

## Docker and Wheel Details

Find specific builds for different hardware and versions:

### Python Packages

[See available Python Packages](https://github.com/pytorch/xla#python-packages)

### Docker

[See available Docker images](https://github.com/pytorch/xla#docker)

## Contributing and Getting Help

Contribute to the project by following the [contribution guide](CONTRIBUTING.md).  For questions and feedback, please file an issue on [Github](https://github.com/pytorch/xla/issues).

## Disclaimer

This repository is jointly operated and maintained by Google, Meta and a
number of individual contributors listed in the
[CONTRIBUTORS](https://github.com/pytorch/xla/graphs/contributors) file. For
questions directed at Meta, please send an email to opensource@fb.com. For
questions directed at Google, please send an email to
pytorch-xla@googlegroups.com. For all other questions, please open up an issue
in this repository [here](https://github.com/pytorch/xla/issues).

## Additional Reads

*   [Performance debugging on Cloud TPU
  VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor
  intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU
  VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with
  FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

* [OpenXLA](https://github.com/openxla)
* [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
* [JetStream](https://github.com/google/JetStream-pytorch)