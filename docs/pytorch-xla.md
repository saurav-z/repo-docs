# PyTorch/XLA: Accelerate Your PyTorch Models with XLA on TPUs and GPUs

**Supercharge your PyTorch deep learning models with the power of XLA (Accelerated Linear Algebra) on Cloud TPUs and GPUs!**  [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA seamlessly integrates the [PyTorch deep learning framework](https://pytorch.org/) with the [XLA deep learning compiler](https://www.tensorflow.org/xla) and [Cloud TPUs](https://cloud.google.com/tpu/), enabling faster training and inference. Try it for free on a single Cloud TPU VM with [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

## Key Features

*   **TPU and GPU Acceleration:**  Run your PyTorch models on Cloud TPUs and GPUs for significant performance gains.
*   **Seamless Integration:** Easy integration with your existing PyTorch code, requiring minimal changes.
*   **XLA Compiler Optimization:** Leverages the XLA compiler for optimized execution on supported hardware.
*   **Kaggle Compatibility:** Get started quickly with readily available Kaggle notebooks.
*   **Comprehensive Documentation:** Access extensive guides and tutorials to help you get started and troubleshoot.

## Getting Started

### Installation

Follow the instructions below for a quick setup and installation.

**Note:**  Ensure you use a supported Python version (3.8 - 3.13).

#### TPU Installation

##### Stable Build:

```bash
pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
pip install 'torch_xla[pallas]'  # Optional: If you're using custom kernels
```

##### Nightly Build:

**As of 07/16/2025 and starting from Pytorch/XLA 2.8 release, PyTorch/XLA will provide nightly and release wheels for Python 3.11 to 3.13**

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

#### C++11 ABI Builds

**As of 03/18/2025 and starting from Pytorch/XLA 2.7 release, C++11 ABI builds are the default and we no longer provide wheels built with pre-C++11 ABI.**

For the legacy 2.6 wheels you can install them like so:

```bash
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

#### GPU Installation

GPU support is available.  Please check the [Available Docker Images and Wheels](#available-docker-images-and-wheels) section for specific installation instructions based on your CUDA and Python versions.

### Getting Your Model Running

*   **Single Process:** Make these modifications to your training loop:
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
        model.to('xla')  # Move model parameters to XLA device
        train(model, training_data, ...)
        ...
    ```

*   **Multi-Process:** Implement this:
    ```python
    import torch_xla
    import torch_xla.core.xla_model as xm

    def _mp_fn(index):
        ...
        model.to('xla')

        for inputs, labels in train_loader:
            with torch_xla.step():
                inputs, labels = inputs.to('xla'), labels.to('xla')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                xm.optimizer_step(optimizer) # Uses the 'xm' XLA model
    if __name__ == '__main__':
        torch_xla.launch(_mp_fn, args=()) # Auto detects the world size
    ```

*   **Distributed Data Parallel (DDP):** Modify your code like this:

    ```python
    import torch.distributed as dist
    import torch_xla
    import torch_xla.distributed.xla_backend

    def _mp_fn(rank):
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

For more detailed instructions, see the [Getting Started](#getting-started) section.

### Examples

Explore these example notebooks to get started quickly:

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Documentation and Guides

*   **Latest Release Documentation:** [https://pytorch.org/xla](https://pytorch.org/xla)
*   **Master Branch Documentation:** [https://pytorch.org/xla/master](https://pytorch.org/xla/master)
*   **API Guide:** [API_GUIDE.md](API_GUIDE.md)

## Detailed Documentation Map:

Find the resources to work with PyTorch/XLA:

*   [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): XLA concepts, troubleshooting, PJRT, Eager Mode and Dynamic Shapes.
*   [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): `GPU` and `TPU` documents.
*   [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): Performance features: `AMP`, `DDP`, `Dynamo`, Fori loop, `FSDP`, quantization, recompilation, and `SPMD`
*   [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed torch, pallas, scan, stable hlo, and triton.
*   [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Setting up PyTorch for development, and guides for lowering operations.
*   PJRT plugins:
    -   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    -   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
    -   [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

## Reference Implementations

*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes): Examples for LLM and diffusion models

## Available Docker Images and Wheels

### Python Packages
PyTorch/XLA releases starting with version r2.1 will be available on PyPI. You can now install the main build with `pip install torch_xla`. To also install the Cloud TPU plugin corresponding to your installed `torch_xla`, install the optional `tpu` dependencies after installing the main build with

```
pip install torch_xla[tpu]
```

GPU release builds and GPU/TPU nightly builds are available in our public GCS bucket.

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

To run on [compute instances with
GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).

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

## Troubleshooting

Encountering issues?  Consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Get Involved

We welcome feedback and contributions from the community!  Please submit issues, bug reports, and feature requests on our [GitHub repository](https://github.com/pytorch/xla).  See the [contribution guide](CONTRIBUTING.md) for more details.

## Stay Connected

*   **For Meta-related questions:** opensource@fb.com
*   **For Google-related questions:** pytorch-xla@googlegroups.com
*   **General questions and issues:** [GitHub Issues](https://github.com/pytorch/xla/issues)

## Additional Resources

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)

**[Visit the PyTorch/XLA GitHub Repository](https://github.com/pytorch/xla)** for more details.