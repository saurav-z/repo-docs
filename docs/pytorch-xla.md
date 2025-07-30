# PyTorch/XLA: Accelerate Your PyTorch Models with Cloud TPUs and GPUs

**Supercharge your PyTorch deep learning workflows with PyTorch/XLA, enabling blazing-fast training and inference on Cloud TPUs and GPUs!** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a powerful Python package that seamlessly integrates the [PyTorch deep learning framework](https://pytorch.org/) with the [XLA deep learning compiler](https://www.tensorflow.org/xla), unlocking the potential of [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs for accelerated model training and deployment.  Get started for free on a single Cloud TPU VM with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

Explore our [Kaggle notebooks](https://github.com/pytorch/xla/tree/master/contrib/kaggle) to quickly learn the ropes:

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Key Features

*   **TPU and GPU Acceleration:** Harness the power of Cloud TPUs and GPUs for faster training and inference.
*   **Easy Integration:** Seamlessly integrates with PyTorch, requiring minimal code changes to leverage XLA.
*   **Optimized Performance:** Leverages the XLA compiler for optimized execution on target hardware.
*   **Distributed Training Support:** Enables scaling your models across multiple TPUs or GPUs.
*   **Comprehensive Documentation:**  Extensive documentation and tutorials to get you started quickly.

## Installation

### TPU Installation

Install PyTorch/XLA on a new TPU VM with the following steps.  Ensure you're using a supported Python version (3.8 to 3.11).

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

**Nightly Builds (for bleeding-edge features and improvements):**

**Note:** Starting from PyTorch/XLA 2.8, nightly and release wheels will be available for Python versions 3.11-3.13.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU Installation

GPU release builds and GPU/TPU nightly builds are available in our public GCS bucket. Refer to the tables below for the wheel links.

**Note:** You need to install the appropriate CUDA and Python versions corresponding to your GPU configuration.

### C++11 ABI Builds

**As of PyTorch/XLA 2.7, C++11 ABI builds are the default.**

**Example - C++11 ABI for PyTorch/XLA 2.6 (Python 3.10):**

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

**C++11 ABI Docker Image:**

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11 /bin/bash
```

### GPU Wheels
```
| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (Python 3.12) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
| nightly (Python 3.13) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
```
### Docker
NOTE: Since PyTorch/XLA 2.7, all builds will use the C++11 ABI by default
```
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
```
To use the above dockers, please pass `--privileged --net host --shm-size=16G` along. Here is an example:
```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```
<br/>

```
| Version | GPU CUDA 12.6 Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` |
```

<br/>

```
| Version | GPU CUDA 12.4 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.4` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.4` |
```

<br/>

```
| Version | GPU CUDA 12.1 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.1` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.1` |
| 2.3 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_cuda_12.1` |
| 2.2 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1` |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_12.1` |
| nightly | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1` |
| nightly at date | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1_YYYYMMDD` |
```

<br/>

```
| Version | GPU CUDA 11.8 + Docker |
| --- | ----------- |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_11.8` |
| 2.0 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.0_3.8_cuda_11.8` |
```

## Github Doc Map

Find in-depth information and guides within our repository:
*   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): Core concepts, troubleshooting, PJRT, eager mode, and dynamic shapes.
*   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): GPU and TPU accelerator documentation.
*   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): Performance optimization: AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed torch, Pallas, scan, stable HLO, and Triton.
*   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): PyTorch development setup and guides for lowering operations.
*   PJRT plugins:
    *   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    *   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): Torchax documentation.
    *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): Torchax examples.

## Getting Started

### Single Process Training

To adapt your existing training loop:

```diff
+import torch_xla

 def train(model, training_data, ...):
   ...
   for inputs, labels in train_loader:
+    with torch_xla.step():
       inputs, labels = training_data[i]
+      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()

+  torch_xla.sync()
   ...

 if __name__ == '__main__':
   ...
+  # Move the model paramters to your XLA device
+  model.to('xla')
   train(model, training_data, ...)
   ...
```

### Multi-Process Training

To adapt your existing training loop:

```diff
-import torch.multiprocessing as mp
+import torch_xla
+import torch_xla.core.xla_model as xm

 def _mp_fn(index):
   ...

+  # Move the model paramters to your XLA device
+  model.to('xla')

   for inputs, labels in train_loader:
+    with torch_xla.step():
+      # Transfer data to the XLA device. This happens asynchronously.
+      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
-      optimizer.step()
+      # `xm.optimizer_step` combines gradients across replicas
+      xm.optimizer_step(optimizer)

 if __name__ == '__main__':
-  mp.spawn(_mp_fn, args=(), nprocs=world_size)
+  # torch_xla.launch automatically selects the correct world size
+  torch_xla.launch(_mp_fn, args=())
```

### DistributedDataParallel

```diff
 import torch.distributed as dist
-import torch.multiprocessing as mp
+import torch_xla
+import torch_xla.distributed.xla_backend

 def _mp_fn(rank):
   ...

-  os.environ['MASTER_ADDR'] = 'localhost'
-  os.environ['MASTER_PORT'] = '12355'
-  dist.init_process_group("gloo", rank=rank, world_size=world_size)
+  # Rank and world size are inferred from the XLA device runtime
+  dist.init_process_group("xla", init_method='xla://')
+
+  model.to('xla')
+  ddp_model = DDP(model, gradient_as_bucket_view=True)

-  model = model.to(rank)
-  ddp_model = DDP(model, device_ids=[rank])

   for inputs, labels in train_loader:
+    with torch_xla.step():
+      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = ddp_model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()

 if __name__ == '__main__':
-  mp.spawn(_mp_fn, args=(), nprocs=world_size)
+  torch_xla.launch(_mp_fn, args=())
```

Additional information is available at [PyTorch.org](http://pytorch.org/xla/). See the [API Guide](API_GUIDE.md) for best practices.

## Tutorials and Resources

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU Guide](docs/gpu.md)

## Reference Implementations

The [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) repo provides examples for training and serving LLMs and diffusion models.

## Troubleshooting

For help debugging and optimizing, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md).

## Feedback and Contribution

We welcome your feedback!  Please file issues on [our GitHub](https://github.com/pytorch/xla) for questions, bug reports, and feature requests.  See the [contribution guide](CONTRIBUTING.md) for contributing.

## Additional Readings

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)