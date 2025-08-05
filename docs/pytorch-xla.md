# PyTorch/XLA: Accelerate Your Deep Learning Models with TPUs and GPUs

[PyTorch/XLA](https://github.com/pytorch/xla) seamlessly integrates the PyTorch deep learning framework with Google Cloud TPUs and GPUs, unlocking significant performance gains for your models.

[![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

**Key Features:**

*   **TPU and GPU Acceleration:** Run your PyTorch models efficiently on Cloud TPUs and GPUs using the XLA compiler.
*   **Simplified Integration:** Easily integrate PyTorch/XLA into your existing training loops with minimal code changes.
*   **Performance Optimization:** Leverage XLA's advanced compilation and optimization techniques for faster training and inference.
*   **Comprehensive Documentation:** Benefit from detailed documentation, tutorials, and guides to get you started quickly.
*   **Open Source and Collaborative:** Contribute to and benefit from a vibrant open-source community.

## Installation

PyTorch/XLA offers flexible installation options for both stable and nightly builds on TPUs and GPUs.

**Important Note:** Please ensure you are using a supported Python version (3.8-3.13 for stable builds as of July 16, 2024, with Python 3.11-3.13 wheels in development for nightly builds)

### TPU Installation

#### Stable Build
```bash
pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
pip install 'torch_xla[pallas]' # Optional: for custom kernels
```
#### Nightly Build

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU Installation
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U torch_xla[gpu] -f https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/index.html
```

### C++11 ABI builds

By default, PyTorch/XLA builds use the C++11 ABI.

## Getting Started

Refer to these quickstart guides:
*   [Cloud TPU VM
    quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice
    quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)

### Code Modifications

Here are examples of minimal code changes to enable your existing model:
*   **Single process**
```python
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

*   **Multi process**

```python
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

*   **Multi process with `DistributedDataParallel`**

```python
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

## Documentation and Resources

*   **Official Documentation:** [PyTorch.org](http://pytorch.org/xla/)
*   **Latest Release Documentation:** [https://pytorch.org/xla](https://pytorch.org/xla)
*   **Master Branch Documentation:** [https://pytorch.org/xla/master](https://pytorch.org/xla/master)
*   **API Guide:** [API_GUIDE.md](API_GUIDE.md)
*   **GitHub Doc Map:**
    *   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): Learning resources.
    *   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): Accelerator documents.
    *   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): Performance-related docs.
    *   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): Feature documentation.
    *   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Contribution guides.
    *   PJRT plugins:
        *   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
        *   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
    *   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
        *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

### Tutorials

*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

### Reference Implementations

*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)

## Available Docker Images and Wheels

### Python packages

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

### Docker
```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```

## Troubleshooting

Find help on improving performance in the [troubleshooting
guide](docs/source/learn/troubleshoot.md).

## Contributing

We welcome your contributions! Please see the [contribution guide](CONTRIBUTING.md).

## Community and Support

*   **Issue Tracker:** Report issues and ask questions on the [GitHub Issues](https://github.com/pytorch/xla/issues).
*   **Feedback:** We value your feedback!

## Disclaimer

This project is jointly maintained by Google, Meta, and other contributors. Contact pytorch-xla@googlegroups.com for Google-related questions, opensource@fb.com for Meta-related questions, and the GitHub issues for general inquiries.