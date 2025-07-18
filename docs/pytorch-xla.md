# PyTorch/XLA: Accelerate Your Deep Learning with TPUs and GPUs

**Supercharge your PyTorch models with the power of Google Cloud TPUs and GPUs using PyTorch/XLA!**

[Go to the PyTorch/XLA GitHub Repository](https://github.com/pytorch/xla)

[![GitHub Actions Status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

PyTorch/XLA is a Python package that seamlessly integrates the [PyTorch deep learning framework](https://pytorch.org/) with the [XLA deep learning compiler](https://www.tensorflow.org/xla), unlocking high-performance training on [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs.

**Key Features:**

*   **TPU and GPU Acceleration:** Train your PyTorch models significantly faster on Cloud TPUs and GPUs.
*   **Easy Integration:** Simple changes to your existing PyTorch code enable TPU/GPU acceleration.
*   **Free Trial:** Get started quickly with free access via [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338).
*   **XLA Compiler:** Leverages the XLA compiler for optimized performance.
*   **Comprehensive Documentation:** Extensive guides, tutorials, and examples to get you started.

**Get Started:**

1.  **Install:**
    *   **Stable Build for TPU VM:**

        ```bash
        pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
        ```

    *   **Nightly Build for TPU VM:**

        ```bash
        pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
        # Edit `cp310-cp310` to fit your desired Python version as needed
        pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
          -f https://storage.googleapis.com/libtpu-wheels/index.html
        ```
    *   **GPU Build (refer to the available wheels):**

        ```bash
        # Example (CUDA 12.6 + Python 3.10):
        pip install torch==2.7.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
        pip install 'torch_xla[cuda]'  # or  'torch_xla[gpu]'
        ```
        (Check available wheels for other CUDA and Python versions)
2.  **Modify Your Training Loop:**
    *   **Single Process:** Integrate `torch_xla.step()` and `.to('xla')` for seamless TPU/GPU utilization.
    *   **Multi-Process:** Utilize `torch_xla.launch` and `xm.optimizer_step` for distributed training.

    Example code snippets for single and multi-process training are included in the Getting Started section of the original README.

3.  **Explore Tutorials:**
    *   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
    *   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
    *   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    *   [GPU guide](docs/gpu.md)
    *   And other useful guides within the repository (see the Github Doc Map section below)

**C++11 ABI builds**
As of Pytorch/XLA 2.7 release, C++11 ABI builds are the default.

To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```sh
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```
**Docker**
NOTE: Since PyTorch/XLA 2.7, all builds will use the C++11 ABI by default

### GitHub Doc Map
| Doc Source  | Contents |
| ----------- | ----------- |
| [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn) | Learning concepts, troubleshooting, pjrt, eager mode, and dynamic shape |
| [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators) |  References to GPU and TPU accelerator documents |
| [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf) |  Documentation about performance specific aspects of PyTorch/XLA such as: AMP, DDP, Dynamo, Fori loop, FSDP, quantization, recompilation, and SPMD |
| [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features) |  Documentation on distributed torch, pallas, scan, stable hlo, and triton |
| [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute) |  Documents on setting up PyTorch for development, and guides for lowering operations |
| PJRT plugins  | CPU, CUDA |
| [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs) |  Torchax documents |
| [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples) |  Torchax examples |

**Available Docker Images and Wheels:**
See the original README for a comprehensive list of available Docker images and wheel files based on version, CUDA version, Python version and hardware target.

**Troubleshooting:**
Consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

**Get Involved:**
*   **Provide Feedback:** File issues on GitHub for questions, bug reports, and feature requests.
*   **Contribute:** See the [contribution guide](CONTRIBUTING.md) for guidelines.

**Additional Resources:**
*   [PyTorch.org](http://pytorch.org/xla/)
*   [Documentation for the latest release](https://pytorch.org/xla)
*   [Documentation for master branch](https://pytorch.org/xla/master)

**Related Projects:**
*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)