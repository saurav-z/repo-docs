# PyTorch/XLA: Accelerate Your PyTorch Models with Cloud TPUs and GPUs

**Supercharge your deep learning workflows by leveraging the power of XLA and Cloud TPUs and GPUs, enabling faster training and inference for your PyTorch models.** [![GitHub Actions status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pytorch/xla)

[Explore the PyTorch/XLA Repository](https://github.com/pytorch/xla)

PyTorch/XLA is a Python package that seamlessly integrates the [PyTorch](https://pytorch.org/) deep learning framework with the [XLA (Accelerated Linear Algebra) deep learning compiler](https://www.tensorflow.org/xla) and Cloud TPUs, enabling users to harness the power of specialized hardware for accelerated training and inference. You can also use PyTorch/XLA on GPU's. Experience the speed boost firsthand with free access on a single Cloud TPU VM through [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

**Key Features:**

*   **TPU and GPU Acceleration:** Run your PyTorch models on Cloud TPUs and GPUs for significant performance gains.
*   **Seamless Integration:** Easy to integrate with existing PyTorch code with minimal modifications.
*   **XLA Compilation:** Leverages the XLA compiler for optimized execution on target hardware.
*   **Distributed Training Support:** Supports multi-process training on multiple TPU/GPU devices.

**Getting Started:**

Check out these example notebooks to start.

*   [Stable Diffusion with PyTorch/XLA 2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
*   [Distributed PyTorch/XLA Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Installation

**Important:**  Ensure you have a compatible Python version (3.8 to 3.13) for your PyTorch/XLA installation.

### TPU

#### Stable Build

```bash
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.7.0 'torch_xla[tpu]==2.7.0'
```

#### Nightly Build

**Note:** Starting with PyTorch/XLA 2.8, nightly and release wheels for Python 3.11 to 3.13 are available.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### GPU

```bash
pip install torch_xla -f https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl
```

or

```bash
pip install torch_xla -f https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl
```

## Github Doc Map

Explore the comprehensive documentation within the repository:

*   [Learning Resources](https://github.com/pytorch/xla/tree/master/docs/source/learn):  XLA concepts, troubleshooting, PJRT, eager mode, and dynamic shapes.
*   [Accelerator Guides](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): Information for `GPU` and `TPU` devices.
*   [Performance Optimization](https://github.com/pytorch/xla/tree/master/docs/source/perf): Guides on performance improvements, including `AMP`, `DDP`, `Dynamo`, `Fori loop`, `FSDP`, and more.
*   [Advanced Features](https://github.com/pytorch/xla/tree/master/docs/source/features): Information on distributed training, Pallas, Scan, and other advanced features.
*   [Contribution Guidelines](https://github.com/pytorch/xla/tree/master/docs/source/contribute): Development setup and guides for lowering operations.
*   **PJRT Plugins:**
    *   [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
    *   [CUDA](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md)
*   [torchax Documentation](https://github.com/pytorch/xla/tree/master/torchax/docs)
    *   [torchax Examples](https://github.com/pytorch/xla/tree/master/torchax/examples)

## Basic Usage

### Single Process

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

## Distributed Data Parallel (DDP) with Multi-Processing

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

## Documentation and Tutorials

*   [API Guide](API_GUIDE.md) for best practices.
*   [Latest Release Documentation](https://pytorch.org/xla)
*   [Master Branch Documentation](https://pytorch.org/xla/master)
*   [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
*   [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
*   [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
*   [GPU guide](docs/gpu.md)

## Reference Implementations

*   [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) for LLM and diffusion model examples.

## Available Packages

### Python Packages

*   Install with `pip install torch_xla` after installing the core `torch_xla` package, then install the  `tpu` package: `pip install torch_xla[tpu]`

**Nightly Builds:** Use the nightly builds for the latest features and improvements. Include the date in the version as shown below.

```bash
pip3 install torch==2.9.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev20250423-cp310-cp310-linux_x86_64.whl
```

**Package Links:** Find the latest package links in the original README.

### Docker Images

*   **TPU VM:** `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_tpuvm` (and older versions)
*   **GPU (CUDA):** `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_cuda_12.6` (and older versions)

**Running Docker Images:** Use `--privileged --net host --shm-size=16G` when running Docker images.

## Troubleshooting

Refer to the [troubleshooting guide](docs/source/learn/troubleshoot.md) for debugging and optimization tips.

## Contribute

*   [Contribution Guide](CONTRIBUTING.md)

## Contact and Support

*   For general questions, bug reports, and feature requests, please file an issue on [GitHub](https://github.com/pytorch/xla/issues).
*   For questions directed at Meta, please send an email to opensource@fb.com.
*   For questions directed at Google, please send an email to pytorch-xla@googlegroups.com.

## Further Reading

*   [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
*   [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
*   [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
*   [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)

## Related Projects

*   [OpenXLA](https://github.com/openxla)
*   [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
*   [JetStream](https://github.com/google/JetStream-pytorch)