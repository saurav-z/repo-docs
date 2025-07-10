# PyTorch/XLA: Accelerate Deep Learning with Cloud TPUs and GPUs

**Supercharge your PyTorch models by seamlessly integrating with Google Cloud TPUs and GPUs using the XLA deep learning compiler.** ([View on GitHub](https://github.com/pytorch/xla))

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that empowers you to run your PyTorch models on both Cloud TPUs and GPUs, leveraging the power of the XLA (Accelerated Linear Algebra) deep learning compiler. This results in faster training and inference, allowing you to experiment more effectively and scale your deep learning projects. You can even try it for free on a single Cloud TPU VM with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

**Key Features:**

*   **TPU and GPU Acceleration:** Run PyTorch models on Cloud TPUs and GPUs for significantly improved performance.
*   **XLA Compilation:** Utilizes the XLA compiler to optimize your models for maximum efficiency on supported hardware.
*   **Easy Integration:** Simple integration with your existing PyTorch code, with minimal code changes required.
*   **Kaggle Compatibility:** Get started quickly with free access on Kaggle.
*   **Distributed Training:** Supports multi-process training to accelerate training across multiple accelerators.

## Getting Started

To quickly see PyTorch/XLA in action, explore these Kaggle notebooks:

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Installation

Follow these steps to get PyTorch/XLA up and running:

### TPU Installation

**Stable Builds:**

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

**Nightly Builds:**

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU Installation
See the table below for wheel links to install for GPU:

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp39-cp39-linux_x86_64.whl` |
| nightly (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |

### C++11 ABI Builds
Starting with PyTorch/XLA 2.7, C++11 ABI builds are the default. For more information, see the original README.

## Github Doc Map
Navigate the different documentation across the repo:
*   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): Learning the concepts associated with XLA, troubleshooting, pjrt, eager mode, and dynamic shape.
*   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): References to `GPU` and `TPU` accelerator documents.
*   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): Documentation on performance specific aspects of PyTorch/XLA such as: `AMP`, `DDP`, `Dynamo`, Fori loop, `FSDP`, quantization, recompilation, and `SPMD`
*   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): Documentation on distributed torch, pallas, scan, stable hlo, and triton.
*   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Documents on setting up PyTorch for development, and guides for lowering operations.
*   PJRT plugins:
    *   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    *   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
    *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

## Getting Started

### Simple single process

To update your exisitng training loop, make the following changes:

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

### Multi processing

To update your existing training loop, make the following changes:

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

If you're using `DistributedDataParallel`, make the following changes:


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

## Guides and Documentation

*   [Documentation for the latest release](https://pytorch.org/xla)
*   [Documentation for master branch](https://pytorch.org/xla/master)

## Tutorials and Resources

*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Reference Implementations

Explore the [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) repository for examples of training and serving LLMs and diffusion models.

## Available Packages and Docker Images

### Python Packages

PyTorch/XLA releases are available on PyPI.  Install the main build using `pip install torch_xla`. To also install the Cloud TPU plugin, use `pip install torch_xla[tpu]`

[See the original README for wheel links.](#available-docker-images-and-wheels)

### Docker Images

NOTE: Since PyTorch/XLA 2.7, all builds will use the C++11 ABI by default
[See the original README for docker links.](#available-docker-images-and-wheels)

## Troubleshooting

If you encounter issues, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Feedback and Contribution

The PyTorch/XLA team welcomes your feedback!  Please submit issues on this GitHub repository.  See the [contribution guide](CONTRIBUTING.md) for details on how to contribute.

## Disclaimer

This project is jointly operated and maintained by Google, Meta, and individual contributors.

## Additional Reads

* [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
* [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
* [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
* [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)