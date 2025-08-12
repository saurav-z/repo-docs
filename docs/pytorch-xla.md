# PyTorch/XLA: Accelerate Your PyTorch Models with XLA on TPUs and GPUs

**PyTorch/XLA enables you to seamlessly run your PyTorch models on Cloud TPUs and GPUs, offering significant performance gains.**

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that integrates the [XLA (Accelerated Linear Algebra) deep learning compiler](https://www.tensorflow.org/xla) with the [PyTorch deep learning framework](https://pytorch.org/), allowing you to leverage the power of [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs for faster training and inference. You can also get started with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

**Key Features:**

*   **TPU and GPU Acceleration:**  Run PyTorch models on Cloud TPUs and GPUs for enhanced performance.
*   **XLA Compiler Integration:**  Utilizes the XLA compiler for efficient execution.
*   **Easy Integration:**  Simple modifications to your existing PyTorch code to utilize XLA.
*   **Comprehensive Documentation:**  Detailed guides and tutorials to get you started.
*   **Active Community:**  Benefit from a supportive community and active development.

Explore these [Kaggle notebooks](https://github.com/pytorch/xla/tree/master/contrib/kaggle) for practical examples:

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Installation

### TPU Installation

Follow these steps to install the stable or nightly build on your TPU VM:

**Prerequisites:** Ensure you have a Python version supported by PyTorch/XLA (Python 3.8-3.11).

**Stable Build Installation**

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'

# Optional: if you're using custom kernels, install pallas dependencies
pip install 'torch_xla[pallas]'
```

**Nightly Build Installation** (For Python 3.11-3.13 and beyond)

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI Builds

From PyTorch/XLA 2.7, C++11 ABI builds are the default.

## Github Doc Map

Find comprehensive documentation to help you get the most out of PyTorch/XLA:

*   [Learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): Concepts, troubleshooting, PJRT, eager mode, and dynamic shapes.
*   [Accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): GPU and TPU accelerator documentation.
*   [Performance](https://github.com/pytorch/xla/tree/master/docs/source/perf): AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   [Features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed torch, Pallas, scan, stable HLO, and Triton.
*   [Contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute):  Setup for development and guides for lowering operations.
*   PJRT Plugins: [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md), [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documentation
    * [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

## Getting Started

*   **Single Process:**  Train on a single GPU/TPU.
*   **Multi-Process:**  Utilize multiple GPU/TPUs.
*   **SPMD:** Single Program, Multiple Data (SPMD)

Refer to the [SPMD guide](https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_basic.md) for in-depth information.

### Single Process Example

Incorporate these changes into your existing training loop:

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

### Multi-Process Example

Update your existing training loop with the following adjustments:

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

For further information on PyTorch/XLA, please visit [PyTorch.org](http://pytorch.org/xla/). Refer to the [API Guide](API_GUIDE.md) for best practices.

**Documentation:**

*   [Latest Release Documentation](https://pytorch.org/xla)
*   [Master Branch Documentation](https://pytorch.org/xla/master)

## PyTorch/XLA Tutorials

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU Guide](docs/gpu.md)

## Reference Implementations

Explore examples for training and serving LLM and diffusion models on the [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) repository.

## Available Docker Images and Wheels

Access pre-built packages and containers to easily get started:

### Python Packages

PyTorch/XLA releases starting with version r2.1 are available on PyPI. Install the main build with `pip install torch_xla`. Install the optional `tpu` dependencies with `pip install torch_xla[tpu]`. GPU and nightly builds are available in our public GCS bucket.

**Wheel Downloads:**

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (Python 3.12) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
| nightly (Python 3.13) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |

**Nightly Build Install Example:**
```bash
pip3 install torch==2.9.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev20250423-cp310-cp310-linux_x86_64.whl
```

**Older Versions (See Original README for comprehensive list)**

### Docker

Use the following commands to use the docker images, and use `--privileged --net host --shm-size=16G` along.

**TPU VM Docker Images:**

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

**Example Docker Run:**

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```

**GPU Docker Images (See Original README for comprehensive list):**

| Version | GPU CUDA 12.6 Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` |

## Troubleshooting

For guidance on resolving performance issues, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md).

## Providing Feedback

We welcome your feedback! Please open an issue on [GitHub](https://github.com/pytorch/xla/issues) for questions, bug reports, and feature requests.

## Contributing

Contribute to the project by following the [contribution guide](CONTRIBUTING.md).

## Disclaimer

This project is jointly maintained by Google, Meta, and individual contributors.

*   Meta: opensource@fb.com
*   Google: pytorch-xla@googlegroups.com
*   General: Open an issue [here](https://github.com/pytorch/xla/issues)

## Additional Resources

*   [Performance Debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy Tensor Intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling Deep Learning Workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)

**[Back to Top](#top)**