# PyTorch/XLA: Accelerate Your Deep Learning with Cloud TPUs and GPUs

**Supercharge your PyTorch models with the power of Google Cloud TPUs and GPUs using PyTorch/XLA!** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

[View the original repository on GitHub](https://github.com/pytorch/xla)

PyTorch/XLA is a Python package that seamlessly integrates the PyTorch deep learning framework with the XLA (Accelerated Linear Algebra) deep learning compiler, enabling high-performance training and inference on Cloud TPUs and GPUs.

**Key Features:**

*   **TPU and GPU Acceleration:** Leverage the computational power of Cloud TPUs and GPUs to significantly speed up your model training and inference.
*   **Easy Integration:** Simple integration with existing PyTorch code with minimal changes.
*   **XLA Compiler:** Utilize the XLA compiler for optimized execution on target hardware.
*   **Distributed Training:** Support for distributed training across multiple TPUs or GPUs for faster model development.
*   **Comprehensive Documentation:** Access detailed guides and tutorials for getting started, performance optimization, and troubleshooting.
*   **Active Community:** Benefit from a supportive community and collaborative development.

## Installation

### TPU Installation

Follow these steps to get started with PyTorch/XLA on Cloud TPUs.

*   **Supported Python Versions:**  Python 3.8 to 3.13 are supported.

**Stable Build Installation:**

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.8.0 'torch_xla[tpu]==2.8.0'

# Optional: if you're using custom kernels, install pallas dependencies
pip install 'torch_xla[pallas]'
```
**Nightly Build Installation:**
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU Installation

*  Please see the available wheels for GPU installation here:
  `https://github.com/pytorch/xla#available-docker-images-and-wheels`
  
### C++11 ABI builds
**C++11 ABI builds are the default. No longer provide wheels built with pre-C++11 ABI.**

  For more information:
  `https://github.com/pytorch/xla#cpp11-abi-builds`

## Key Resources

### GitHub Doc Map

Explore a wealth of documentation within our GitHub repository.

*   [Learning Resources](https://github.com/pytorch/xla/tree/master/docs/source/learn):  Concepts, troubleshooting, PJRT, eager mode, and dynamic shape.
*   [Accelerator Guides](https://github.com/pytorch/xla/tree/master/docs/source/accelerators):  GPU and TPU-specific documents.
*   [Performance Optimization](https://github.com/pytorch/xla/tree/master/docs/source/perf): AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   [Advanced Features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed torch, Pallas, Scan, stable HLO, and Triton.
*   [Contribution Guide](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Setting up development, and lowering operations.
*   PJRT Plugins:
    *   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    *   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
    *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

## Getting Started

### Single Process

Adapt your training loop with minimal changes for single-process execution.

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

Modify your code for multi-process TPU/GPU utilization.

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

Adapt your DistributedDataParallel code.

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

## Tutorials & Resources

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU Guide](docs/gpu.md)

## Reference Implementations

Explore example models and training recipes in the [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) repository.

## Available Builds

### Python Packages

PyTorch/XLA packages are available on PyPI. You can install the main build with `pip install torch_xla`.  Install the TPU plugin with `pip install torch_xla[tpu]`.

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (Python 3.12) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
| nightly (Python 3.13) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |

#### Use nightly build

```
pip3 install torch==2.9.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev20250423-cp310-cp310-linux_x86_64.whl
```
For older versions of wheels, see: `https://github.com/pytorch/xla#available-docker-images-and-wheels`

### Docker Images

*  See Docker image installation here:
  `https://github.com/pytorch/xla#available-docker-images-and-wheels`

## Troubleshooting

Refer to the [troubleshooting guide](docs/source/learn/troubleshoot.md) for assistance in debugging and optimizing your models.

## Community & Contributions

We welcome your feedback and contributions!  Please submit issues, bug reports, and feature requests through our GitHub repository.  See the [contribution guide](CONTRIBUTING.md) for details on contributing.

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)