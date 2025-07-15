# PyTorch/XLA: Accelerate Your PyTorch Models with Cloud TPUs and GPUs

**Supercharge your PyTorch deep learning models by leveraging the power of Google Cloud TPUs and GPUs with the PyTorch/XLA library.**  [Get started with PyTorch/XLA](https://github.com/pytorch/xla) to unlock significant performance gains for your machine learning projects.

**Key Features:**

*   **TPU Acceleration:** Seamlessly integrate with Cloud TPUs using the XLA (Accelerated Linear Algebra) compiler for blazing-fast training and inference.
*   **GPU Support:** Run your PyTorch models on GPUs using the XLA compiler, optimizing performance.
*   **Ease of Use:** Simple integration with your existing PyTorch code with minimal modifications, accelerating your model with a few lines of code.
*   **Free Trial:** Try PyTorch/XLA for free on a single Cloud TPU VM with Kaggle!
*   **Distributed Training:**  Support for distributed training across multiple TPUs/GPUs for scaling up model complexity and dataset sizes.
*   **C++11 ABI Builds:**  Leverage C++11 ABI builds for improved lazy tensor tracing performance, particularly beneficial for complex models.
*   **Comprehensive Documentation:** Extensive documentation and tutorials to guide you through installation, usage, and optimization.

## Getting Started

### Installation

#### TPU

Follow the steps below to install the PyTorch/XLA for Cloud TPU VMs:

**Stable Build:**

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
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

#### GPU

The installation varies by CUDA and Python versions, please check the documentation for the most up-to-date installation instructions:

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp39-cp39-linux_x86_64.whl` |
| nightly (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl` |

### Using Single Process

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

### Using Multi Process

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

## Documentation and Resources

*   **Official Documentation:** [PyTorch.org](http://pytorch.org/xla/)
*   **API Guide:** [API_GUIDE.md](API_GUIDE.md)
*   **Tutorials:**
    *   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
    *   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
    *   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    *   [GPU Guide](docs/gpu.md)

## Troubleshooting and Support

*   **Troubleshooting Guide:** [docs/source/learn/troubleshoot.md](docs/source/learn/troubleshoot.md)
*   **Feedback:**  File an issue on [GitHub](https://github.com/pytorch/xla/issues) to report bugs, request features, or ask questions.

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)