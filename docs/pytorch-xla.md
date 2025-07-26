# PyTorch/XLA: Accelerate Your PyTorch Models with TPUs and GPUs

**Supercharge your deep learning workflows by connecting PyTorch with Google Cloud TPUs and GPUs using the XLA compiler.** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions)

This repository provides the Python package that bridges the gap between the [PyTorch](https://pytorch.org/) deep learning framework and the [XLA deep learning compiler](https://www.tensorflow.org/xla), enabling high-performance training and inference on [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs. Get started today with free access on Kaggle! ([original repo](https://github.com/pytorch/xla))

## Key Features

*   **TPU Acceleration:** Seamlessly run your PyTorch models on Google Cloud TPUs for significantly faster training.
*   **GPU Support:**  Leverage GPUs with XLA to optimize your model performance.
*   **Easy Integration:** Simple changes to your existing PyTorch code are all that are required.
*   **Performance Optimization:**  Utilize XLA's compiler to optimize your model's performance.
*   **Comprehensive Documentation:** Access detailed guides and tutorials for getting started.

## Getting Started

### Installation

**TPU Installation:**

Install the stable release using pip.  Ensure you are using a supported Python version (3.8 to 3.13).

```bash
# Install with the correct Python version and venv setup
# python3.11 -m venv py311
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

**Nightly Builds:**

For the latest features and bug fixes, install the nightly build:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

**GPU Installation:**

```bash
pip install torch_xla[cuda]
```

**C++11 ABI Builds:**

For improved performance with lazy tensor tracing, install C++11 ABI wheels.

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

### Tutorials and Guides

*   **Cloud TPU VM quickstart:** Learn how to get started with Cloud TPUs.
*   **Cloud TPU Pod slice quickstart:** Quickly get up and running with TPU Pod slices.
*   **Profiling on TPU VM:** Learn how to profile your models on TPUs.
*   **GPU guide:** Comprehensive guide to running PyTorch/XLA on GPUs.

## Getting Started Guide

### Simple Single Process

To update your existing training loop, follow these steps:

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
## Documentation and Community

*   **[Official Documentation](https://pytorch.org/xla):** Comprehensive documentation for the latest release.
*   **[API Guide](API_GUIDE.md):** Best practices for writing networks that run on XLA devices.
*   **[GitHub Repository](https://github.com/pytorch/xla):**  Find source code, examples, and contribute.
*   **[Troubleshooting Guide](docs/source/learn/troubleshoot.md):**  Address potential issues and optimize your networks.
*   **[Contribution Guide](CONTRIBUTING.md):** Learn how to contribute to PyTorch/XLA.

## Example Notebooks
*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)