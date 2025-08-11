# PyTorch/XLA: Accelerate Your Deep Learning with TPUs and GPUs

**Supercharge your PyTorch models with the power of XLA compilers on TPUs and GPUs, enabling faster training and inference.**  [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml)

[PyTorch/XLA](https://github.com/pytorch/xla) seamlessly integrates the PyTorch framework with the XLA (Accelerated Linear Algebra) deep learning compiler, unlocking the potential of Google Cloud TPUs and GPUs for accelerated deep learning workloads. Get started today with free access via Kaggle!

**Key Features:**

*   **TPU and GPU Acceleration:** Utilize the computational power of TPUs and GPUs to significantly speed up model training and inference.
*   **Integration with PyTorch:** Leverages the familiar PyTorch ecosystem and API.
*   **XLA Compiler:** Benefit from the XLA deep learning compiler for optimized performance.
*   **Cloud TPU Compatibility:** Directly integrate with Google Cloud TPUs.
*   **Easy Installation:** Simple installation process with pip.
*   **Comprehensive Documentation:** Access detailed guides, tutorials, and examples.
*   **Active Community:** Benefit from ongoing development and community support.

## Installation

Follow these steps to get PyTorch/XLA up and running.

### TPU Installation

**Supported Python Versions:** Builds are available for Python 3.8 to 3.13.

1.  **Create a virtual environment (recommended):**
    ```bash
    # For venv
    # python3.11 -m venv py311
    # For conda
    # conda create -n py311 python=3.11
    ```
2.  **Install the stable PyTorch/XLA package:**

    ```bash
    pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
    ```

3.  **(Optional) Install Pallas dependencies for custom kernels:**

    ```bash
    pip install 'torch_xla[pallas]'
    ```

### Nightly Builds

**Note:** Starting from PyTorch/XLA 2.8 release, nightly and release wheels will be provided for Python 3.11 to 3.13.

To install the nightly build in a new TPU VM:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU Installation

For running on compute instances with GPUs, refer to the [detailed GPU guide](docs/gpu.md)

### C++11 ABI Builds

**Note:** C++11 ABI builds are the default starting from PyTorch/XLA 2.7.

## Github Doc Map

Navigate the PyTorch/XLA repository with these helpful documentation links:

*   [Learning Resources](https://github.com/pytorch/xla/tree/master/docs/source/learn): Learn XLA concepts and troubleshooting.
*   [Accelerator Guides](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): GPU and TPU specifics.
*   [Performance Optimization](https://github.com/pytorch/xla/tree/master/docs/source/perf): Improve performance with AMP, DDP, and more.
*   [Feature Documentation](https://github.com/pytorch/xla/tree/master/docs/source/features): Distributed training, Pallas, and other features.
*   [Contribution Guidelines](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Set up your environment and contribute.
*   [PJRT Plugins](https://github.com/pytorch/xla/tree/master/plugins): CPU and CUDA plugins.
*   [Torchax Docs](https://github.com/pytorch/xla/tree/master/torchax/docs): Documentation for the `torchax` library.

## Getting Started

Easily integrate PyTorch/XLA into your training workflows with these code snippets.

### Single Process Training

1.  **Import the XLA library:**

    ```python
    import torch_xla
    ```
2.  **Move the model parameters to the XLA device:**

    ```python
    model.to('xla')
    ```
3.  **Wrap the training loop:**

    ```python
    with torch_xla.step():
        # ... your training code ...
    ```

4.  **Sync after the loop:**

    ```python
    torch_xla.sync()
    ```

### Multi-Process Training

1.  **Import necessary libraries:**

    ```python
    import torch_xla
    import torch_xla.core.xla_model as xm
    ```
2.  **Move model parameters to the XLA device:**

    ```python
    model.to('xla')
    ```
3.  **Wrap the training loop:**

    ```python
    with torch_xla.step():
        inputs, labels = inputs.to('xla'), labels.to('xla')
        # ... your training code ...
        xm.optimizer_step(optimizer) # Use xm.optimizer_step for multi-process
    ```

4.  **Launch training with torch_xla.launch:**

    ```python
    torch_xla.launch(_mp_fn, args=())
    ```

**For DistributedDataParallel**, make the necessary changes as described in the original README.

## Tutorials and Resources

*   [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod Slice Quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Reference Implementations

*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes): Examples for training and serving LLM and diffusion models.

## Available Docker Images and Wheels

Find the latest Docker images and Python wheels for PyTorch/XLA in the table format provided in the original README.

## Troubleshooting

If you encounter issues, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging tips and optimization strategies.

## Providing Feedback

Your feedback is valuable! Report any issues or suggestions on the [GitHub Issues page](https://github.com/pytorch/xla/issues).

## Contributing

Contribute to PyTorch/XLA development by following the guidelines in the [contribution guide](CONTRIBUTING.md).

## Disclaimer

This project is jointly maintained by Google, Meta, and community contributors. Contact them at the provided email addresses or by opening an issue on GitHub.

## Additional Reads

*   [Performance Debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy Tensor Intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling Deep Learning Workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)