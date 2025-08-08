# PyTorch/XLA: Accelerate Deep Learning with Cloud TPUs and GPUs

**Supercharge your PyTorch models with the power of Google Cloud TPUs and GPUs, boosting performance and reducing training times!** [Explore the PyTorch/XLA repository on GitHub](https://github.com/pytorch/xla).  [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that seamlessly integrates the PyTorch deep learning framework with the XLA (Accelerated Linear Algebra) compiler, enabling you to leverage the performance of both Cloud TPUs and GPUs. This powerful combination allows you to accelerate your machine learning workloads, experiment with larger models, and achieve faster training cycles.

**Key Features:**

*   **Cloud TPU Integration:** Effortlessly run your PyTorch models on Google Cloud TPUs for significant speedups.
*   **GPU Support:** Utilize GPUs to further optimize performance.
*   **XLA Compiler:** Benefits from the XLA compiler's optimization capabilities for efficient execution.
*   **Kaggle Integration:**  Experiment with PyTorch/XLA for free using Kaggle notebooks.
*   **Easy to use**: Simplified integration with XLA.

## Getting Started

### Installation

Follow these steps to install and get started with PyTorch/XLA.

**Stable Build Installation (TPU)**
**Note**: Make sure you're using a supported Python version.

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'

# Optional: if you're using custom kernels, install pallas dependencies
pip install 'torch_xla[pallas]'
```

**Nightly Build Installation (TPU)**
**Note**: Starting with Pytorch/XLA 2.8 release, PyTorch/XLA will provide nightly and release wheels for Python 3.11 to 3.13.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

**C++11 ABI Builds**
**Note**: C++11 ABI builds are the default starting with Pytorch/XLA 2.7 release, and wheels built with pre-C++11 ABI are no longer provided.

To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

**C++11 ABI Docker Image**

```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```
### Basic Usage Guides
#### Single Process
```python
import torch_xla

 def train(model, training_data, ...):
   ...
   for inputs, labels in train_loader:
    with torch_xla.step():
       inputs, labels = training_data[i]
       inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()

   torch_xla.sync()
   ...

 if __name__ == '__main__':
   ...
   # Move the model paramters to your XLA device
   model.to('xla')
   train(model, training_data, ...)
   ...
```
#### Multi Process
```python
import torch_xla
import torch_xla.core.xla_model as xm

 def _mp_fn(index):
   ...

  # Move the model paramters to your XLA device
  model.to('xla')

   for inputs, labels in train_loader:
    with torch_xla.step():
      # Transfer data to the XLA device. This happens asynchronously.
      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
      # `xm.optimizer_step` combines gradients across replicas
      xm.optimizer_step(optimizer)

 if __name__ == '__main__':
  # torch_xla.launch automatically selects the correct world size
  torch_xla.launch(_mp_fn, args=())
```
#### Multi Processing with DDP
```python
 import torch.distributed as dist
import torch_xla
import torch_xla.distributed.xla_backend

 def _mp_fn(rank):
   ...

  # Rank and world size are inferred from the XLA device runtime
  dist.init_process_group("xla", init_method='xla://')

  model.to('xla')
  ddp_model = DDP(model, gradient_as_bucket_view=True)

   for inputs, labels in train_loader:
    with torch_xla.step():
      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = ddp_model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()

 if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
```
### Example Notebooks
* [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
* [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Documentation and Resources

*   **Official Documentation:** [PyTorch.org](http://pytorch.org/xla/)
*   **API Guide:** [API_GUIDE.md](API_GUIDE.md)
*   **Comprehensive User Guides:**
    *   [Documentation for the latest release](https://pytorch.org/xla)
    *   [Documentation for master branch](https://pytorch.org/xla/master)
*   **Tutorials:**
    *   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
    *   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
    *   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    *   [GPU guide](docs/gpu.md)
*   **Useful Reading Materials**
    * [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
    * [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
    * [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
    * [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)
## GitHub Documentation Map
-   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): Learn concepts associated with XLA, troubleshooting, pjrt, eager mode, and dynamic shape.
-   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): `GPU` and `TPU` accelerator documents.
-   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): Performance specific aspects of PyTorch/XLA.
-   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): Documentation on distributed torch, pallas, scan, stable hlo, and triton.
-   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Documents on setting up PyTorch for development.
-   PJRT plugins:
    -   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    -   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
-   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
    -   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples
## Docker Images and Wheels
### Wheels
*   **PyPI:** Starting with version r2.1, PyTorch/XLA releases are available on PyPI. Install the main build with `pip install torch_xla`. To install the Cloud TPU plugin, install the optional `tpu` dependencies after installing the main build with:
    ```bash
    pip install torch_xla[tpu]
    ```
*   **GPU/TPU Nightly Builds:** [Available in our public GCS bucket.](#available-docker-images-and-wheels)
### Docker
To use the dockers, please pass `--privileged --net host --shm-size=16G` along.
#### TPU Docker Images
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

#### GPU Docker Images
| Version | GPU CUDA Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.4` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.4` |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.1` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.1` |
| 2.3 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_cuda_12.1` |
| 2.2 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1` |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_12.1` |
| nightly | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1` |
| nightly at date | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1_YYYYMMDD` |
### GPU Instances
Run on [compute instances with GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).

## Troubleshooting

If you encounter performance issues, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Feedback and Contribution

The PyTorch/XLA team welcomes your feedback and contributions!  Please file issues on GitHub for questions, bug reports, and feature requests. Refer to the [contribution guide](CONTRIBUTING.md) for details on how to contribute.

## Disclaimer

This project is maintained by Google, Meta, and individual contributors. Contact pytorch-xla@googlegroups.com for Google-related inquiries or opensource@fb.com for Meta-related inquiries. For other questions, please open an issue.

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)