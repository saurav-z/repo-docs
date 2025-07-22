# PyTorch/XLA: Accelerate Your Deep Learning with TPUs and GPUs

**Supercharge your PyTorch models by leveraging the power of Google Cloud TPUs and GPUs with PyTorch/XLA!** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that seamlessly integrates the [PyTorch deep learning framework](https://pytorch.org/) with the [XLA (Accelerated Linear Algebra) deep learning compiler](https://www.tensorflow.org/xla) and [Cloud TPUs](https://cloud.google.com/tpu/). This allows you to accelerate your model training and inference, particularly for large-scale deep learning workloads.  You can try it right now, for free, on a single Cloud TPU VM with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

Explore practical examples in our [Kaggle notebooks](https://github.com/pytorch/xla/tree/master/contrib/kaggle):

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Key Features

*   **TPU and GPU Acceleration:**  Run your PyTorch models efficiently on Google Cloud TPUs and GPUs.
*   **XLA Compilation:**  Leverage the XLA compiler for optimized performance.
*   **Easy Integration:**  Simple integration with existing PyTorch code.
*   **Kaggle Integration:**  Get started quickly with pre-configured Kaggle notebooks.
*   **Comprehensive Documentation:** Extensive documentation to guide you through setup, usage, and troubleshooting.

## Installation

Choose your installation method based on your target accelerator (TPU or GPU) and Python version.  Make sure you have a compatible Python version (3.8 - 3.13) to get started.

### TPU Installation

**Stable Build (Recommended):**

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
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU Installation

Wheels are available for CUDA-enabled GPUs.

**Stable Build (CUDA 12.6 + Python 3.10 example):**
```bash
pip install torch_xla==2.7.0 -f https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/index.html
```

**Nightly Build (CUDA 12.6 + Python 3.10 example):**
```bash
pip install torch==2.7.0+cu126 --index-url https://download.pytorch.org/whl/nightly/cpu
pip install torch_xla==2.9.0.dev20250423 -f https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/index.html
```
Consult the [Available docker images and wheels](#available-docker-images-and-wheels) section below for specific wheel links for various CUDA versions and Python versions.

## Getting Started

PyTorch/XLA supports single-process and multi-process training configurations. The following snippets will help you update your exisiting training loops:

### Simple Single Process

To modify your existing training loop, use the following changes:

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

### Multi-Processing

To update your existing training loop for multi-processing:

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

For `DistributedDataParallel`, apply these modifications:

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

## GitHub Doc Map
Explore comprehensive documentation within the repository:

*   [Learning Resources](https://github.com/pytorch/xla/tree/master/docs/source/learn): Learn concepts, troubleshooting tips, PJRT, and eager mode.
*   [Accelerator Guides](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): Find specific references for GPU and TPU accelerators.
*   [Performance Optimization](https://github.com/pytorch/xla/tree/master/docs/source/perf): Discover performance-related aspects like AMP, DDP, Dynamo, and more.
*   [Feature Documentation](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed training, Pallas, and SPMD guides.
*   [Contribution Guide](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Instructions for development setup and lowering operations.
*   PJRT Plugins:
    *   [CPU Plugin](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    *   [CUDA Plugin](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [Torchax Documentation](https://github.com/pytorch/xla/tree/master/torchax/docs): Documents on torchax and examples.

## Available Docker Images and Wheels

### Python Packages
PyTorch/XLA releases starting with version r2.1 are available on PyPI.

```bash
pip install torch_xla[tpu]
```

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (Python 3.12) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
| nightly (Python 3.13) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |

### Docker

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```

For older versions, consult the original README.

## Troubleshooting

If you experience performance issues, refer to the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Resources and Support

*   [Documentation for the latest release](https://pytorch.org/xla)
*   [Documentation for master branch](https://pytorch.org/xla/master)
*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

**For support and feedback, please file an issue on [GitHub](https://github.com/pytorch/xla/issues).**

## Contributing

We welcome contributions! Please review the [contribution guide](CONTRIBUTING.md).

## Disclaimer
This project is maintained by Google, Meta, and community contributors.

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)

**[Back to Top](#top)**