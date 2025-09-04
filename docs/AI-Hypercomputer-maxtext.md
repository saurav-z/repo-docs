# MaxText: Train and Deploy High-Performance LLMs with Ease

**Maximize your LLM performance** with MaxText, a high-performance, open-source LLM framework built in pure Python/Jax, optimized for training and inference on Google Cloud TPUs and GPUs. [(GitHub Repo)](https://github.com/AI-Hypercomputer/maxtext)

## Key Features

*   **High Performance:** Achieves impressive Model Flops Utilization (MFU) for faster training and inference.
*   **Scalability:** Designed to scale from single hosts to massive clusters of TPUs and GPUs.
*   **Open Source:**  Built on open-source principles, encouraging community contributions and customization.
*   **Flexible Hardware Support:** Supports both Google Cloud TPUs and GPUs.
*   **Training and Inference:** Provides comprehensive support for both model training and deployment.
*   **Model Support:**  Out-of-the-box support for leading LLM architectures, including:
    *   Llama 2, Llama 3, Llama 4 (Maverick & Scout)
    *   Mistral & Mixtral family
    *   Gemma, Gemma 2, Gemma 3
    *   DeepSeek
    *   Qwen3 Dense and MoE family

## Announcements

*   **[August 13, 2025]**: Qwen3 MoE family of models supported.
*   **[July 27, 2025]**: Updated TFLOPS/s calculation and attention optimizations.
*   **[July 16, 2025]**: Repository restructuring proposed.
*   **[July 11, 2025]**: Multi-Token Prediction (MTP) training support.
*   **[June 25, 2025]**: DeepSeek R1-0528 variant support.
*   **[April 24, 2025]**: Llama 4 Maverick models support.
*   **[April 14, 2025]**: Llama 4 Scout models support.
*   **[April 7, 2025]**: Modular imports and API changes for `train.py`.
*   **[April 2, 2025]**: DeepSeek v3-0324 variant support.
*   **[March 24, 2025]**: DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]**: Gemma 3 support (4B, 12B, 27B).
*   **[February, 2025]**: (Preview) MaxText Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Begin your MaxText journey with our [Getting Started](getting_started/First_run.md) guide.

Leverage pre-built guides to train and deploy supported open models:

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

Explore the `end_to_end` directory and [unit tests](.github/workflows/RunTests.yml) for further insight.

## Runtime Performance Results

Detailed performance results are available in [MaxText/configs/README.md](MaxText/configs/README.md).

### TPU v5p

| No. of params | Accelerator Type | TFLOP/chip/sec | Model flops utilization (MFU) |
|---|---|---|---|
| 32B | v5p-128 | 3.28e+02 | 67.76% |
| 64B | v5p-128 | 3.23e+02 | 70.31% |
| 128B | v5p-256 | 3.15e+02 | 68.68% |
| 128B | v5p-512 | 3.15e+02 | 68.53% |
| 256B | v5p-1024 | 3.16e+02 | 68.82% |
| 512B | v5p-1024 | 2.94e+02 | 63.99% |
| 1024B | v5p-2048 | 2.49e+02 | 64.05% |
| 1024B | v5p-4096 | 2.97e+02 | 64.80% |
| 1160B | v5p-7680 | 2.95e+02 | 64.27% |
| 1160B | v5p-12288 | 3.04e+02 | 66.23% |

### TPU v5e

See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText offers compelling performance compared to other LLM implementations. It draws inspiration from [MinGPT](https://github.com/karpathy/minGPT) and [NanoGPT](https://github.com/karpathy/nanoGPT), while addressing their limitations by supporting a greater number of industry standard models and scalability for large scale deployments. Additionally, MaxText offers significantly improved MFU compared to other open-source alternatives.

Compared to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), MaxText offers competitive performance, however, MaxText leverages pure Python and the XLA compiler, in contrast to the hybrid Python/CUDA approach employed by Megatron-LM.

Similar to [Pax](https://github.com/google/paxml), MaxText prioritizes high-performance and scalability in Jax. However, while Pax emphasizes extensive configuration options, MaxText focuses on a simple and directly modifiable codebase.

## Features and Diagnostics

### Collect Stack Traces

To debug SPMD jobs on accelerators, use these settings in `MaxText/configs/base.yml`:

1.  `collect_stack_trace: True` enables stack trace collection on errors, and `collect_stack_trace: False` disables it.
2.  `stack_trace_to_cloud: True` stores traces in cloud logging; `stack_trace_to_cloud: False` displays them on the console.
3.  `stack_trace_interval_seconds` controls the frequency of trace collection.

For further details, see the [cloud-tpu-diagnostics](https://pypi.org/project/cloud-tpu-diagnostics) package.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile your training for optimal performance and faster startup times.  Install `jax[tpu]` to compile.

##### Example AOT 1: Compile ahead of time basics

```bash
# Run the below on a single machine, e.g. a CPU
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

##### Example AOT 2: Save compiled function, then load and run it

**Step 1: Run AOT and save compiled function**

```bash
# Run the below on a single machine, e.g. a CPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
# Run the below on each host of the target hardware, e.g. each host on 2 slices of v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

#### GPU Support

1.  Single GPU host is required to run AoT compilation, but a single GPU host can compile a program for a larger cluster of the same hardware.
2.  For [A3 Cloud GPUs](https://cloud.google.com/compute/docs/gpus#h100-gpus), the maximum "slice" size is a single host, and the `compile_topology_num_slices` parameter represents the number of A3 machines to precompile for.

##### Example

**Step 1: Run AOT and save compiled function**

```bash
# Run the below on a single A3 machine
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
# Run the below on each of the 4 target A3 hosts.
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for automatic log uploads.

### Monitor Goodput of Your Workload

Consult the [user guide](getting_started/Monitor_Goodput.md) for goodput monitoring.