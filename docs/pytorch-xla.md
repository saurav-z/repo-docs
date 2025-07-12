# PyTorch/XLA: Accelerate Your Deep Learning with Cloud TPUs and GPUs

**Supercharge your PyTorch models by leveraging the power of Google Cloud TPUs and GPUs with the PyTorch/XLA library!** ([Original Repo](https://github.com/pytorch/xla))

[![GitHub Actions Status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that seamlessly integrates the [PyTorch deep learning framework](https://pytorch.org/) with the [XLA (Accelerated Linear Algebra) deep learning compiler](https://www.tensorflow.org/xla), enabling efficient execution on both [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs. This integration allows you to accelerate your PyTorch models for faster training and inference.

**Key Features:**

*   **TPU Acceleration:** Run your PyTorch models on Google Cloud TPUs for significant performance gains.
*   **GPU Support:** Utilize GPUs for faster training and inference.
*   **XLA Compilation:** Leverage the XLA compiler for optimized performance.
*   **Ease of Use:** Simple integration with existing PyTorch code.
*   **Active Community:** Benefit from a supportive community and extensive documentation.
*   **Free Trial:** Experiment with PyTorch/XLA on a single Cloud TPU VM through Kaggle.

**Get Started with PyTorch/XLA:**

Explore these example notebooks to get started:

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Installation

### TPU Installation

Follow these steps to install PyTorch/XLA for your environment:

**1. Create a Virtual Environment (Recommended):**

```bash
# Example for Python 3.11
python3.11 -m venv py311
```

**2. Install PyTorch/XLA:**

**Stable Build:**
```bash
pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

**Nightly Build:**

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Replace `cp310-cp310` with your Python version
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI builds

Since PyTorch/XLA 2.7, C++11 ABI builds are the default.

## Github Doc Map

Explore the following docs within the repository:

*   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): XLA concepts, troubleshooting, PJRT, eager mode, and dynamic shape.
*   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): GPU and TPU accelerator documentation.
*   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): Performance aspects: AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed torch, pallas, scan, stable HLO, and triton.
*   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Setting up PyTorch for development, and guides for lowering operations.
*   PJRT plugins: [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md), [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): Torchax documentation.
    *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): Torchax examples.

## Getting Started

Guides are available for both single and multi-process modes. Note:

*   Single process: one Python interpreter controlling a single GPU/TPU at a time
*   Multi process: N Python interpreters are launched, corresponding to N GPU/TPUs
*   SPMD (Single Program, Multiple Data): one Python interpreter controls all N GPU/TPUs on the system. See the [SPMD guide](https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_basic.md).

### Simple Single Process

Update your training loop:

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
+  # Move the model parameters to your XLA device
+  model.to('xla')
   train(model, training_data, ...)
   ...
```

### Multi-Processing

Update your training loop:

```diff
-import torch.multiprocessing as mp
+import torch_xla
+import torch_xla.core.xla_model as xm

 def _mp_fn(index):
   ...

+  # Move the model parameters to your XLA device
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

If using `DistributedDataParallel`:

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

## Documentation

*   [PyTorch.org](http://pytorch.org/xla/) - Additional information.
*   [API Guide](API_GUIDE.md) - Best practices.
*   [Latest Release Documentation](https://pytorch.org/xla)
*   [Master Branch Documentation](https://pytorch.org/xla/master)

## PyTorch/XLA Tutorials

*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Reference Implementations

*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)

## Available Docker Images and Wheels

### Python Packages

PyTorch/XLA releases starting with version r2.1 are available on PyPI.
Install the main build with `pip install torch_xla` and the optional `tpu` dependency: `pip install torch_xla[tpu]`.

**GPU Wheels:**

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp39-cp39-linux_x86_64.whl` |
| nightly (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |

**Use Nightly Build:**
```bash
pip3 install torch==2.8.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev20250423-cp310-cp310-linux_x86_64.whl
```

Refer to the documentation for older versions and wheels.

### Docker Images

NOTE: Since PyTorch/XLA 2.7, all builds use the C++11 ABI by default

**TPU Docker Images:**

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

Run with `--privileged --net host --shm-size=16G`.

**GPU Docker Images:**

| Version | GPU CUDA 12.6 Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` |

Refer to the documentation for older versions and Docker images.

## Troubleshooting

Consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Providing Feedback

We welcome your feedback!  File an issue on this GitHub repository for questions, bug reports, or feature requests.

## Contributing

See the [contribution guide](CONTRIBUTING.md).

## Disclaimer

This project is jointly maintained by Google, Meta, and individual contributors.

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)