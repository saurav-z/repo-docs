# PyTorch/XLA: Accelerate Your PyTorch Models with TPUs and GPUs

**Supercharge your PyTorch models with the power of Google Cloud TPUs and GPUs using PyTorch/XLA!** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a powerful Python package that bridges the gap between the PyTorch deep learning framework and Google Cloud TPUs (Tensor Processing Units) and GPUs, enabling faster training and inference for your models.  Get started for free on a single Cloud TPU VM with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

Access the code repository for detailed information and examples:  [PyTorch/XLA GitHub](https://github.com/pytorch/xla).

## Key Features

*   **TPU and GPU Acceleration:** Leverage the massive parallel processing capabilities of TPUs and GPUs to significantly speed up your PyTorch training and inference.
*   **Seamless Integration:**  Easy integration with existing PyTorch code.  Modify your code with minimal changes to utilize XLA devices.
*   **XLA Compiler:** Utilizes the XLA (Accelerated Linear Algebra) deep learning compiler for optimized performance.
*   **Cloud TPU Support:**  Directly supports Google Cloud TPUs, allowing you to scale your models for demanding workloads.
*   **GPU Support:** Enables acceleration on compatible GPUs.
*   **Nightly Builds and Release Wheels:** Access to pre-built wheels for the latest features and bug fixes.
*   **Comprehensive Documentation:** Extensive documentation to guide you through installation, usage, and optimization.
*   **Active Community:** Benefit from a supportive community and open-source collaboration.

## Getting Started

### Installation

Choose the appropriate installation method based on your target device (TPU or GPU) and desired Python version. Detailed instructions and links to specific wheel files are provided below.  Note that support for Python versions 3.8 through 3.13 is included.

#### TPU Installation

1.  **Prerequisites:** Ensure you have a Python environment set up (e.g., using `venv` or `conda`).
2.  **Stable Build:** Install the stable PyTorch/XLA build.  Replace the placeholder versions with your desired PyTorch/XLA versions:
    ```bash
    # Example (replace 2.7.0 with your desired version)
    pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
    ```
3.  **Nightly Build:** Install the latest nightly build. Edit the install commands to match your desired Python version:
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    # Replace cp310-cp310 with your desired Python version as needed
    pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
      -f https://storage.googleapis.com/libtpu-wheels/index.html
    ```

#### GPU Installation

For GPU installations, please refer to the section on available wheels below.

#### C++11 ABI builds
C++11 ABI builds are the default. To install, you'll need to find the wheel that matches your cuda version and desired python version.

```bash
# Python 3.10 example
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

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

## GitHub Documentation Map

*   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): XLA concepts, troubleshooting, PJRT, eager mode, and dynamic shapes.
*   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): GPU and TPU accelerator documents.
*   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): Performance-specific documentation: AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD.
*   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed torch, pallas, scan, stable hlo, and triton.
*   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Setting up PyTorch for development and guides for lowering operations.
*   PJRT plugins:
    *   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    *   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
    *   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

## Tutorials and Resources

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU Guide](docs/gpu.md)
*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)

## Available Wheels and Docker Images

### Python Packages

PyTorch/XLA releases starting with version r2.1 are available on PyPI.
```bash
pip install torch_xla
pip install torch_xla[tpu]
```

#### Available wheels

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
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
 For earlier versions, refer to the expanded "older versions" section in the original README.

### Docker

#### Cloud TPU VM Docker Images
```
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```
For earlier versions, refer to the expanded "older versions" section in the original README.

## Troubleshooting

Refer to the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Contributing and Feedback

We welcome your contributions! Please see the [contribution guide](CONTRIBUTING.md).  For questions, bug reports, or feature requests, please file an issue on this GitHub repository.

## Additional Information

*   [PyTorch.org](http://pytorch.org/xla/)
*   [API Guide](API_GUIDE.md)
*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)

---