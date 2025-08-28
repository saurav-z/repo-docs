# MaxText: High-Performance, Scalable LLM Training and Inference

**MaxText empowers researchers and developers to train and deploy large language models (LLMs) with exceptional speed and efficiency using pure Python/JAX.**

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/RunTests.yml)

## Overview

MaxText is a **high-performance, highly scalable, and open-source** LLM framework designed for both **training and inference** of large language models. Built with pure Python/JAX and optimized for Google Cloud TPUs and GPUs, MaxText delivers impressive performance and scales seamlessly from single-host setups to massive clusters. Its simplicity and reliance on the power of JAX and the XLA compiler allow for "optimization-free" operation, enabling developers to focus on innovation.

MaxText serves as a robust foundation for ambitious LLM projects, both in research and production environments.  Experiment with out-of-the-box functionality and then customize MaxText to meet specific needs.

**Key Features:**

*   **TPU & GPU Support:** Optimized for Google Cloud TPUs (v5e, v5p) and GPUs (A3).
*   **Training and Inference:** Comprehensive support for both LLM training and deployment.
*   **Model Compatibility:**  Supports a wide range of open-source models, including Llama 2, Llama 3, Llama 4, Mistral, Mixtral, Gemma, Gemma 2, Gemma 3, DeepSeek, Qwen3 Dense and MoE.
*   **Scalability:** Designed to scale from single devices to large clusters.
*   **High Performance:** Achieves high Model Flops Utilization (MFU) on supported hardware.
*   **Modular Imports:**  Uses modular imports, simplifying usage and maintenance.
*   **Ahead-of-Time Compilation (AOT):** Compile your training runs for faster startup and OOM detection, on both TPUs and GPUs.
*   **Diagnostics and Monitoring:** Collect stack traces, and upload logs to Vertex Tensorboard.

## Announcements

*   [August 13, 2025] Support for the Qwen3 MoE family of models, including Qwen3-235B-A22B-Thinking-2507, Qwen3-30B-A3B, and Qwen3-Coder-480B-A35B, in addition to the Qwen3 Dense family.
*   [July 27, 2025] Updated TFLOPS/s calculation to reflect causal and sliding window attention, improving accuracy.
*   [July 16, 2025] Repository restructuring planned for improved organization.  See [RESTRUCTURE.md](RESTRUCTURE.md).
*   [July 11, 2025] Multi-Token Prediction (MTP) training support.
*   [June 25, 2025] DeepSeek R1-0528 variant support.
*   [April 24, 2025] Llama 4 Maverick models support.
*   [April 14, 2025] Llama 4 Scout models support (text-only, 8k context).
*   **[April 7, 2025] 🚨🚨🚨 Modular imports are now supported.  Use `python3 -m MaxText.train MaxText/configs/base.yml run_name=...` instead of `python MaxText/train.py MaxText/configs/base.yml run_name=...`.**
*   [April 2, 2025] DeepSeek v3-0324 variant support.
*   [March 24, 2025] DeepSeek v3 (671B) and v2-Lite (16B) support.
*   [March 12, 2025] Gemma 3 support (4B, 12B, and 27B, text-only).
*   [February, 2025] (Preview): Building MaxText Docker images using the JAX AI Training Images. Learn more [Here](getting_started/Run_MaxText_via_xpk.md)

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get started with MaxText by following the detailed [instructions](getting_started/First_run.md).

MaxText supports training and inference for many open models. Refer to the user guides in the [getting started](getting_started) directory to learn more.

Helpful resources:
*   [Gemma](https://ai.google.dev/gemma): A family of open-weights LLMs by Google DeepMind. Run decode and finetuning using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md). For Gemma 2 and 3, see the scripts in the [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) for checkpoint convertion and decoding.
*   [Llama2](https://llama.meta.com/llama2/): Open-weights LLMs by Meta. Run decode and finetuning using [these instructions](getting_started/Run_Llama2.md).
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Sparse mixture-of-experts (MoE) models by Mistral AI.  Decode and finetune with [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Sparse MoE models by DeepSeek AI. Run pre-training, finetuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Explore the comprehensive end-to-end tests in [end_to_end](end_to_end), run nightly, for an understanding of MaxText's capabilities.  Alternatively, see the continuous [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

Find detailed performance results in [MaxText/configs/README.md](MaxText/configs/README.md).

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

See full run configs in [MaxText/configs/v5e/] as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText draws inspiration from projects such as [MinGPT](https://github.com/karpathy/minGPT) and [NanoGPT](https://github.com/karpathy/nanoGPT), which are elegant standalone GPT implementations in PyTorch. However, MaxText is more complex and supports industry-standard models while scaling to tens of thousands of chips. Ultimately MaxText offers MFU more than three times the [17%](https://twitter.com/karpathy/status/1613250489097027584?cxt=HHwWgIDUhbixteMsAAAA) reported most recently with that codebase, and it's massively scalable and implements a key-value cache for efficient auto-regressive decoding.

MaxText is similar to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), an LLM implementation targeting Nvidia GPUs. The two implementations achieve comparable MFUs.  The difference in the codebases highlights different programming strategies. MaxText is pure Python, relying heavily on the XLA compiler to achieve high performance. By contrast, Megatron-LM is a mix of Python and CUDA, relying on well-optimized CUDA kernels to achieve high performance.

MaxText is also comparable to [Pax](https://github.com/google/paxml). Like Pax, MaxText provides high-performance and scalable implementations of LLMs in Jax. Pax focuses on enabling powerful configuration parameters, enabling developers to change the model by editing config parameters. By contrast, MaxText is a simple, concrete implementation of various LLMs that encourages users to extend by forking and directly editing the source code.

## Features and Diagnostics

### Collect Stack Traces

When running SPMD jobs, capturing stack traces helps identify and troubleshoot issues. Change the values in `MaxText/configs/base.yml`:

1.  Set `collect_stack_trace: True` to enable.  Set `collect_stack_trace: False` to disable.
2.  Set `stack_trace_to_cloud: False` to show on console. `stack_trace_to_cloud: True` will upload to Cloud Logging. Query in Logs Explorer:
    ```
    logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
    jsonPayload.verb="stacktraceanalyzer"
    ```
3.  `stack_trace_interval_seconds` sets the collection frequency (e.g., `600` for every 10 minutes).

Related PyPI package: https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)

Compile training runs ahead of time with `train_compile.py`:

#### TPU Support

*   Compile for a TPU cluster using a CPU or single VM.
*   Flags out-of-memory (OOM) conditions.
*   Saves compilation for faster startup.

##### Example AOT 1: Compile ahead of time basics

```bash
# Run on a single machine, e.g. a CPU
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

##### Example AOT 2: Save compiled function, then load and run it

**Step 1: Run AOT and save compiled function**

```bash
# Run on a single machine, e.g. a CPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
# Run on each host of the target hardware, e.g. each host on 2 slices of v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

#### GPU Support

*   GPU does not support compilation across hardware.
*   For [A3 Cloud GPUs](https://cloud.google.com/compute/docs/gpus#h100-gpus), the maximum "slice" size is a single host.

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

MaxText supports automatic log uploads to a Vertex AI Tensorboard instance. Follow [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

See the [user guide](getting_started/Monitor_Goodput.md) to monitor goodput metrics.

[Back to top](#maxtext-high-performance-scalable-llm-training-and-inference)