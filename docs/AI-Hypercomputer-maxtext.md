# MaxText: High-Performance, Scalable, Open-Source LLM Training and Inference

**Maximize your LLM potential with MaxText, a cutting-edge, open-source framework built for high-performance training and inference on TPUs and GPUs.**

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/RunTests.yml)

## Key Features

*   **High Performance:** Achieves exceptional Model Flops Utilization (MFU) for fast training and inference.
*   **Scalability:** Designed to scale from single-host to very large clusters, supporting thousands of chips.
*   **Open Source:** Built in pure Python/Jax, fostering community contributions and customization.
*   **TPU & GPU Support:** Optimized for Google Cloud TPUs and GPUs.
*   **Training and Inference:** Supports both training and inference workloads.
*   **Model Compatibility:** Supports a wide range of LLMs, including Llama 2/3/4, Mistral/Mixtral, Gemma/Gemma 2/3, and DeepSeek families.

## Overview

MaxText is a powerful, open-source Large Language Model (LLM) framework written in pure Python/Jax, specifically designed for high-performance and scalability on Google Cloud TPUs and GPUs. Built for both training and inference, MaxText leverages the power of Jax and the XLA compiler to deliver exceptional Model Flops Utilization (MFU) and seamlessly scales from single-host to massive clusters. It's an ideal starting point for ambitious LLM projects, providing a flexible foundation for research and production.  Explore the original repository [here](https://github.com/AI-Hypercomputer/maxtext).

## What's New

*   **[July 27, 2024]**:  Attention flop calculations updated to account for causal, sliding window and chunked attention.
*   **[July 16, 2024]**:  Repository restructuring for improved organization and clarity is proposed; see [RESTRUCTURE.md](RESTRUCTURE.md) for feedback.
*   **[July 11, 2024]**:  Multi-Token Prediction (MTP) training support added, enhancing training efficiency.
*   **[June 25, 2024]**:  DeepSeek R1-0528 variant supported.
*   **[April 24, 2024]**: Llama 4 Maverick models are now supported.
*   **[April 14, 2024]**: Llama 4 Scout models are now supported (8k context length).
*   **[April 7, 2024]**: Modular imports supported with associated API change for `train.py`.
*   **[April 2, 2024]**: DeepSeek v3-0324 variant is now supported.
*   **[March 24, 2024]**: DeepSeek v3 (671B) and v2-Lite (16B) support added.
*   **[March 12, 2024]**: Gemma 3: 4B, 12B, and 27B support added.
*   **[February, 2024]**: Preview of MaxText Docker image building using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Start your MaxText journey with the [getting started instructions](getting_started/First_run.md).

### Supported Models & Guides

Explore the following resources to get started:

*   **Gemma:** Fine-tune and decode with [these instructions](end_to_end/tpu/gemma/Run_Gemma.md).  Gemma 2 & 3 have their own scripts in [end_to_end/tpu/gemma2](end_to_end/tpu/gemma2) and [end_to_end/tpu/gemma3](end_to_end/tpu/gemma3).
*   **Llama2:** Fine-tune and decode with [these instructions](getting_started/Run_Llama2.md).
*   **Mixtral:** Fine-tune and decode with [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:** Pre-train, fine-tune, and decode with [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

## Runtime Performance Results

See the performance achieved on TPUs. More details on reproducing these results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

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

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText, heavily influenced by projects like MinGPT and NanoGPT, delivers superior MFU, and is comparable to Megatron-LM, while offering the benefits of pure Python and XLA compilation.

## Features and Diagnostics

### Collect Stack Traces

Enhance debugging by collecting stack traces during SPMD jobs. Configure the following parameters in `MaxText/configs/base.yml`:

1.  `collect_stack_trace: True` to enable trace collection.
2.  `stack_trace_to_cloud: False` to display traces on the console, or `True` to store them in the cloud using a temporary directory and then log the traces to Cloud Logging.
3.  `stack_trace_interval_seconds` sets the trace collection interval.

Package: [cloud-tpu-diagnostics](https://pypi.org/project/cloud-tpu-diagnostics).

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile training runs for faster startup and restart times.

*   TPU Support: Pre-compile on a CPU or single VM to identify OOM issues and save the compiled function for efficient execution. See detailed examples in the original README.
*   GPU Support: A single GPU host can compile for a larger cluster of the same hardware.

### Automatically Upload Logs to Vertex Tensorboard

Integrate with Vertex AI Tensorboard for automatic log uploads. See the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

Track goodput metrics with the [user guide](getting_started/Monitor_Goodput.md).