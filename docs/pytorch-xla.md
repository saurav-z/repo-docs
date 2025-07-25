# PyTorch/XLA: Accelerate Your Deep Learning with TPUs and GPUs

**Supercharge your PyTorch models with PyTorch/XLA, a powerful package that harnesses the XLA deep learning compiler to accelerate training and inference on Cloud TPUs and GPUs.** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla)

[Explore the PyTorch/XLA repository](https://github.com/pytorch/xla) to access this powerful tool!

**Key Features:**

*   **TPU and GPU Acceleration:** Seamlessly run your PyTorch models on Cloud TPUs and GPUs for faster training and inference.
*   **XLA Compilation:** Leverages the XLA (Accelerated Linear Algebra) compiler for optimized performance.
*   **Kaggle Integration:**  Get started quickly with pre-configured environments on Kaggle.
*   **Easy Installation:** Simple installation via pip, with pre-built wheels for various Python versions and hardware configurations.
*   **Comprehensive Documentation:** Extensive documentation and tutorials to guide you through setup, optimization, and troubleshooting.
*   **Active Community:** Benefit from a supportive community and dedicated team, with multiple communication channels including Github Issues and Mailing list

**Get Started with PyTorch/XLA**

You can quickly get started with pre-configured environments on Kaggle:

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

**Installation Guide**

Follow these steps to install PyTorch/XLA:

### TPU

**Stable Build:**

```bash
# Install torch and torch_xla for a specific version.  Replace 2.7.0 with your desired version
pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

**Nightly Build:**

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU

**Available Wheels**

See the table below for wheel links depending on CUDA and Python versions:

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

**Pre-C++11 ABI builds**
*  To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

*   For older versions:
    *  3.9: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl
    *  3.10: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl
    *  3.11: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl

**Docker Images**

*   **TPU VMs:**

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```

*   **GPU CUDA:**

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6 /bin/bash
```

**Getting Started: Code Changes**

Simple single process:

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

Multi processing:

```python
import torch_xla
import torch_xla.core.xla_model as xm

def _mp_fn(index):
   ...

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

**DistributedDataParallel:**

```python
 import torch.distributed as dist
 import torch_xla
 import torch_xla.distributed.xla_backend

def _mp_fn(rank):
   ...

  # Rank and world size are inferred from the XLA device runtime
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

**Documentation and Resources**

*   [Official Documentation](https://pytorch.org/xla)
*   [Latest Release Documentation](https://pytorch.org/xla)
*   [Master Branch Documentation](https://pytorch.org/xla/master)
*   [API Guide](API_GUIDE.md)
*   [GPU guide](docs/gpu.md)

**Troubleshooting and Feedback**

*   If you experience issues, refer to the [troubleshooting guide](docs/source/learn/troubleshoot.md).
*   Provide feedback and report issues on [GitHub](https://github.com/pytorch/xla/issues).

**Additional Resources**

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

**Related Projects**

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)

**Contribute**

Explore the [contribution guide](CONTRIBUTING.md) to help make PyTorch/XLA better.

**Disclaimer**

This project is jointly maintained by Google, Meta, and community contributors.  Reach out to opensource@fb.com for Meta-specific inquiries, pytorch-xla@googlegroups.com for Google-related questions, and use GitHub Issues for all other matters.