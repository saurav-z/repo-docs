# PyTorch/XLA: Accelerate Your Deep Learning with TPUs and GPUs

**Supercharge your PyTorch models with PyTorch/XLA, enabling seamless execution on Cloud TPUs and GPUs.** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla/)

PyTorch/XLA is a Python package that leverages the [XLA deep learning compiler](https://www.tensorflow.org/xla) to connect the [PyTorch deep learning framework](https://pytorch.org/) with [Cloud TPUs](https://cloud.google.com/tpu/) and GPUs.  Get started today with a free single Cloud TPU VM on [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

Explore these example notebooks to see PyTorch/XLA in action:

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Key Features

*   **TPU and GPU Acceleration:** Run your PyTorch models on powerful Cloud TPUs and GPUs for significant performance gains.
*   **Simplified Integration:** Easily integrate PyTorch/XLA into your existing PyTorch training loops with minimal code changes.
*   **Free Trial:** Experiment with PyTorch/XLA on a single Cloud TPU VM for free on Kaggle.
*   **Comprehensive Documentation:** Access detailed guides, tutorials, and API references for seamless development.
*   **Active Community:** Benefit from a vibrant community and responsive support for all your questions and needs.

## Installation

### TPU Installation

**Prerequisites:** Ensure you have Python 3.8 to 3.13 installed, as builds are available for these versions.

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
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU Installation

To run on [compute instances with GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).

For specific installation instructions and available wheels, please refer to the [Available docker images and wheels](#available-docker-images-and-wheels) section.

## Getting Started

### Single Process

Here are the code modifications to integrate into your training loop:

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
    model.to('xla')
    train(model, training_data, ...)
    ...
```

### Multi-Process

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
            xm.optimizer_step(optimizer)

if __name__ == '__main__':
    torch_xla.launch(_mp_fn, args=())
```

### Multi-Process with DistributedDataParallel

```python
import torch.distributed as dist
import torch_xla
import torch_xla.distributed.xla_backend

def _mp_fn(rank):
    ...

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

## Documentation and Resources

*   **[Official PyTorch/XLA Documentation](https://pytorch.org/xla)**
*   **[API Guide](API_GUIDE.md)**
*   **Tutorials:**
    *   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
    *   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
    *   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    *   [GPU guide](docs/gpu.md)

## Available Docker Images and Wheels

*   **Python Packages:**
    *   Install the main build with `pip install torch_xla`.
    *   Install the Cloud TPU plugin with `pip install torch_xla[tpu]`.
    *   Find specific wheel versions for TPUs and GPUs in the table provided in the original README (use the updated links from the original README or the provided tables).
*   **Docker Images:**  Use the provided Docker images for quick setup.  Refer to the original README for specific image tags for different versions, including TPU and GPU configurations.

## Troubleshooting

If you encounter any performance issues or unexpected behavior, consult the [troubleshooting guide](docs/source/learn/troubleshoot.md) for helpful tips.

## Contributing

We welcome contributions!  See the [contribution guide](CONTRIBUTING.md) for details.

## Get Involved

*   [Report Issues](https://github.com/pytorch/xla/issues)
*   [Join the Community](pytorch-xla@googlegroups.com) (for Google-related inquiries)
*   [Contact Meta](opensource@fb.com) (for Meta-related inquiries)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)

**[Explore the PyTorch/XLA Repository](https://github.com/pytorch/xla) to start accelerating your deep learning projects today!**