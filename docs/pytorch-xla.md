# PyTorch/XLA: Accelerate Your Deep Learning with TPUs and GPUs

[PyTorch/XLA](https://github.com/pytorch/xla) seamlessly integrates the PyTorch deep learning framework with the XLA compiler to harness the power of Cloud TPUs and GPUs, enabling faster model training and inference.  [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

**Key Features:**

*   **TPU and GPU Acceleration:** Utilize the XLA compiler to run your PyTorch models on Cloud TPUs and GPUs for significant performance gains.
*   **Easy Integration:** Simple modifications to your existing PyTorch code are all that's needed to start using XLA.
*   **Kaggle Compatibility:** Experiment with PyTorch/XLA directly within Kaggle notebooks, free of charge.
*   **Comprehensive Documentation:** Extensive documentation, tutorials, and examples to help you get started and optimize your models.
*   **Active Community:** Benefit from a supportive community and readily available resources.

## Getting Started

### Installation

*   **TPU:** Install the stable or nightly build with pip. Make sure to select a supported Python version (3.8 - 3.13).

    ```bash
    pip install torch==2.7.0 'torch_xla[tpu]==2.7.0' # Stable - Replace 2.7.0 with desired version.
    ```

    For nightly builds:

    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' -f https://storage.googleapis.com/libtpu-wheels/index.html
    ```

*   **GPU:** Install with pip, selecting the appropriate CUDA version. Check the [available wheels](#available-docker-images-and-wheels) for your CUDA and Python version.

    ```bash
    pip install torch_xla[gpu]  # Or the specific wheel for your CUDA version
    ```

### Quick Start Guides

*   **Single Process:** Adapt your existing training loop for TPU/GPU with minor changes.
*   **Multi-Processing:** Launch N Python interpreters, corresponding to N GPU/TPUs.
*   **SPMD:** Use one Python interpreter to control all N GPU/TPUs (Check the [SPMD guide](https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_basic.md) for more details).

See the sections below for code examples.

## Available Docker Images and Wheels

Choose the appropriate Docker image or wheel for your hardware, CUDA version, and Python version. See detailed tables below.

### Python Packages

Find the wheel for your machine in the links below.

| Version | Cloud GPU VM Wheels |
| --- | ----------- |
| 2.7 (CUDA 12.6 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.7 (CUDA 12.6 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
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

<details>

<summary>older versions</summary>

| Version | Cloud TPU VMs Wheel |
|---------|-------------------|
| 2.7 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.6 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.4 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.3 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.3.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.2 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (XRT + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/xrt/tpuvm/torch_xla-2.1.0%2Bxrt-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.1.0-cp38-cp38-linux_x86_64.whl` |

<br/>

| Version | GPU Wheel |
| --- | ----------- |
| 2.5 (CUDA 12.1 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.1 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.4 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.4/torch_xla-2.5.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.4 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.4/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (CUDA 12.4 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.4/torch_xla-2.5.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.4 (CUDA 12.1 + Python 3.9) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp39-cp39-manylinux_2_28_x86_64.whl` |
| 2.4 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.4 (CUDA 12.1 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.3 (CUDA 12.1 + Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.3.0-cp38-cp38-manylinux_2_28_x86_64.whl` |
| 2.3 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.3.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.3 (CUDA 12.1 + Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.3.0-cp311-cp311-manylinux_2_28_x86_64.whl` |
| 2.2 (CUDA 12.1 + Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.2.0-cp38-cp38-manylinux_2_28_x86_64.whl` |
| 2.2 (CUDA 12.1 + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 + CUDA 11.8 | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/11.8/torch_xla-2.1.0-cp38-cp38-manylinux_2_28_x86_64.whl` |
| nightly + CUDA 12.0 >= 2023/06/27| `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.0/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |

</details>

### Docker

Run the Docker images with appropriate privileges.

NOTE: Since PyTorch/XLA 2.7, all builds will use the C++11 ABI by default
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

To use the above dockers, please pass `--privileged --net host --shm-size=16G` along. Here is an example:
```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```
<br/>

| Version | GPU CUDA 12.6 Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` |


<br/>


| Version | GPU CUDA 12.4 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.4` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.4` |

<br/>


| Version | GPU CUDA 12.1 Docker |
| --- | ----------- |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.1` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.1` |
| 2.3 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_cuda_12.1` |
| 2.2 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1` |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_12.1` |
| nightly | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1` |
| nightly at date | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1_YYYYMMDD` |

<br/>

| Version | GPU CUDA 11.8 + Docker |
| --- | ----------- |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_11.8` |
| 2.0 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.0_3.8_cuda_11.8` |

<br/>

### Code Examples:

These code examples illustrate the minimum changes required to move your PyTorch model to run on the desired device.

#### Single Process

```python
import torch_xla

# ... your model and training setup ...

 def train(model, training_data, ...):
   # ...
   for inputs, labels in train_loader:
     with torch_xla.step():
       inputs, labels = training_data[i]
       inputs, labels = inputs.to('xla'), labels.to('xla') # Move data to XLA device
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()

   torch_xla.sync() # Synchronize XLA devices at the end of the step
   # ...

 if __name__ == '__main__':
   # ...
   model.to('xla') # Move the model to the XLA device
   train(model, training_data, ...)
   # ...
```

#### Multi-Processing

```python
import torch_xla
import torch_xla.core.xla_model as xm

 def _mp_fn(index):
   # ...
   model.to('xla') # Move the model to the XLA device

   for inputs, labels in train_loader:
     with torch_xla.step():
       inputs, labels = inputs.to('xla'), labels.to('xla') # Move data to XLA device
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       xm.optimizer_step(optimizer) # Use xm.optimizer_step for multi-process

 if __name__ == '__main__':
   torch_xla.launch(_mp_fn, args=()) # Launch multi-processing training
```

#### Multi-Processing with Distributed Data Parallel

```python
import torch.distributed as dist
import torch_xla
import torch_xla.distributed.xla_backend

 def _mp_fn(rank):
   dist.init_process_group("xla", init_method='xla://')  # Initialize distributed training
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
   torch_xla.launch(_mp_fn, args=()) # Launch multi-process training
```

### Guides & Resources

*   **Documentation:** [Comprehensive documentation](https://pytorch.org/xla) for the latest release and [master branch](https://pytorch.org/xla/master).
*   **Tutorials:** Explore the Cloud TPU VM quickstart, Cloud TPU Pod slice quickstart, and performance profiling guide.
*   **Reference Implementations:** Explore examples in the [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) repository.
*   **API Guide:** [API Guide](API_GUIDE.md) for best practices.
*   **Troubleshooting:** The [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and performance optimization.
*   **User Guides:** [Documentation for the latest release](https://pytorch.org/xla) and [Documentation for master branch](https://pytorch.org/xla/master)

### Additional Reads

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Contribute and Get Involved

We welcome contributions! Refer to the [contribution guide](CONTRIBUTING.md) to get started.

## Feedback and Support

We're always eager to hear from you! Submit questions, bug reports, and feature requests via issues on [GitHub](https://github.com/pytorch/xla/issues).

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)