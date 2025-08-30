# MaxText: High-Performance LLM Training and Inference in JAX

**Maximize your LLM potential with MaxText, an open-source, high-performance LLM built in pure Python/JAX, optimized for Google Cloud TPUs and GPUs.**

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/RunTests.yml)

## Overview

MaxText is a **high-performance**, **highly scalable**, and **open-source** LLM, written entirely in Python/JAX, designed for efficient **training** and **inference** on Google Cloud TPUs and GPUs. Leveraging the power of JAX and the XLA compiler, MaxText achieves impressive [Model Flops Utilization (MFU)](#runtime-performance-results) and scales seamlessly from single host to massive clusters, all while maintaining a simple and optimization-free design.

Built as a launching pad for ambitious LLM projects, MaxText empowers users to experiment and customize the codebase to meet their unique needs.

We've demonstrated MaxText's capabilities with:
*   [High-performance, well-converging training in int8](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e)
*   [Scaling training to ~51K chips](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e).

**Key Features:**

*   **TPU and GPU Support:** Optimized for Google Cloud hardware.
*   **Training and Inference:** Comprehensive support for both use cases.
*   **Model Compatibility:** Supports a wide range of models, including:
    *   Llama 2, Llama 3, Llama 4
    *   Mistral and Mixtral families
    *   Gemma, Gemma 2, Gemma 3
    *   DeepSeek
    *   Qwen3 (Dense and MoE families)

## Announcements

*   [August 13, 2025] The Qwen3 MoE family of models is now supported. We are starting with Qwen3-235B-A22B-Thinking-2507, Qwen3-30B-A3B and Qwen3-Coder-480B-A35B in addition to our existing Qwen3 Dense family of 0.6B, 4B, 8B, 14B, and 32B models.
*   [July 27, 2025] TFLOPS/s calculation updated to account for causal, sliding window and chunked attention.
*   [July 16, 2025] Restructuring of the MaxText repository for improved organization and clarity. Please review the [proposed structure](RESTRUCTURE.md) and provide feedback.
*   [July 11, 2025] Multi-Token Prediction (MTP) training is now supported!
*   [June 25, 2025] DeepSeek R1-0528 variant is now supported!
*   [April 24, 2025] Llama 4 Maverick models are now supported!
*   [April 14, 2025] Llama 4 Scout models are now supported.
*   **[April 7, 2025] 🚨🚨🚨 Modular imports supported. API change for `train.py`: Invoke the script via `python3 -m MaxText.train MaxText/configs/base.yml run_name=...`. For older API use `python MaxText/train.py MaxText/configs/base.yml run_name=...`.**
*   [April 2, 2025] DeepSeek v3-0324 variant is now supported!
*   [March 24, 2025] DeepSeek v3 (671B) and v2-Lite (16B) support, compatible with TPUs and GPUs.
*   [March 12, 2025] Gemma 3: 4B, 12B, and 27B in text-only formats supported.
*   [February, 2025] (Preview): Maxtext Docker image building with the JAX AI Training Images. Learn more [Here](getting_started/Run_MaxText_via_xpk.md)

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

To begin using MaxText, start with the [getting started](getting_started/First_run.md) guide.

MaxText supports training and inference for various open models. Follow the user guides in the [getting started](getting_started) folder.

**Helpful Guides:**

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma): a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini research and technology. You can run decode and finetuning using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md). For Gemma 2 and 3, use the corresponding [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts for checkpoint convertion and decoding.
*   [Llama2](https://llama.meta.com/llama2/): a family of open-weights Large Language Model (LLM) by Meta. You can run decode and finetuning using [these instructions](getting_started/Run_Llama2.md).
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): a family of open-weights sparse mixture-of-experts (MoE) models by Mistral AI. You can run decode and finetuning using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): a novel family of open-weights sparse MoE models by DeepSeek AI. DeepSeek-V3 features advanced techniques, including Multi-Head Latent Attention (MLA), finer-grained and shared experts, Multi-Token Prediction (MTP), and FP8 mixed precision designed for enhanced efficiency and performance. You can run pre-training, finetuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Explore the full suite of end-to-end tests in [end_to_end](end_to_end). Check the continuous [unit tests](.github/workflows/RunTests.yml) as well.

## Runtime Performance Results

Detailed information on reproducing these results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

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

For 16B, 32B, 64B, and 128B models. See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText draws inspiration from, but differentiates itself from:

*   **MinGPT/NanoGPT:** While elegant, MaxText supports a broader range of models and scales to thousands of chips.
*   **Nvidia/Megatron-LM:** MaxText achieves comparable MFU using pure Python and XLA, while Megatron-LM relies on CUDA kernels.
*   **Pax:** Unlike Pax's configuration-driven approach, MaxText prioritizes a simple, direct implementation encouraging users to extend the code.

## Features and Diagnostics

### Collect Stack Traces

Enable `collect_stack_trace: True` to capture stack traces, helping in debugging SPMD jobs that may hang or crash on accelerators.  Configure `stack_trace_to_cloud: True` to upload traces to Cloud Logging (accessible via Logs Explorer). Set `stack_trace_interval_seconds` to control the frequency of trace collection.

*   View traces in Logs Explorer using the following query:
    ```
    logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
    jsonPayload.verb="stacktraceanalyzer"
    ```
*   Related PyPI package: https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)

The `train_compile.py` tool pre-compiles the `train_step` function for faster startup and restart times on target hardware.

#### TPU Support

*   Use a CPU or a single VM for pre-compilation.
*   Benefits:  OOM detection and faster startup/restart.
*   Install `jax[tpu]` and other dependencies.

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

*   A single GPU host is required for AOT compilation.
*   For A3 Cloud GPUs, the maximum "slice" size is a single host.

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

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to enable automatic log uploads to Vertex AI Tensorboard.

### Monitor Goodput of Your Workload

Refer to the [user guide](getting_started/Monitor_Goodput.md) for instructions on monitoring Goodput metrics.

[**Visit the MaxText GitHub repository**](https://github.com/AI-Hypercomputer/maxtext) to explore the code and contribute.