# MaxText: High-Performance, Scalable LLM Training and Inference in Pure Python/Jax

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxtext)

**Maximize your LLM potential with MaxText, an open-source, high-performance framework built for training and inference on TPUs and GPUs.**

## Key Features

*   **High Performance:** Achieves high Model Flops Utilization (MFU) for efficient training and inference.
*   **Scalable:** Designed to scale from single hosts to massive clusters, supporting training jobs on tens of thousands of chips.
*   **Open Source:** Built with pure Python/Jax, promoting transparency and customization.
*   **TPU & GPU Support:** Optimized for Google Cloud TPUs and GPUs.
*   **Training & Inference:** Comprehensive support for both training and inference workflows.
*   **Model Support:** Includes Llama 2, Llama 3, Mistral/Mixtral, Gemma, Gemma 2, Gemma 3, DeepSeek, and Qwen3 families.

## Overview

MaxText is a cutting-edge LLM framework written in pure Python/Jax, designed for high-performance and scalability on Google Cloud TPUs and GPUs. It's an open-source solution for both training and inference, allowing users to build and experiment with state-of-the-art language models. MaxText simplifies LLM development while delivering exceptional performance.

MaxText has demonstrated impressive results, including [high-performance, well-converging training in int8](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e) and [scaling training to ~51K chips](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e).

## Recent Announcements

*   **[August 13, 2025]** Added support for the Qwen3 MoE family of models, including Qwen3-235B-A22B-Thinking-2507, Qwen3-30B-A3B, and Qwen3-Coder-480B-A35B.
*   **[July 27, 2025]** Updated TFLOPS/s calculation for causal attention.
*   **[July 16, 2025]** Repository restructuring for improved organization.
*   **[July 11, 2025]** Introduced Multi-Token Prediction (MTP) training support.
*   **[June 25, 2025]** DeepSeek R1-0528 variant support.
*   **[April 24, 2025]** Llama 4 Maverick models are now supported!
*   **[April 14, 2025]** Llama 4 Scout models are now supported (text-only).
*   **[April 7, 2025]** Modular imports with API change for `train.py`.
*   **[April 2, 2025]** DeepSeek v3-0324 variant support.
*   **[March 24, 2025]** DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]** Gemma 3 (4B, 12B, and 27B) support.
*   **[February, 2025]** Preview of Maxtext Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get started with MaxText by following the [First Run](getting_started/First_run.md) instructions.

## Supported Models

MaxText supports training and inference of a variety of open-source models.

*   **Gemma:** Run decode and fine-tuning using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md). For Gemma 2 and 3, use the corresponding [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts.
*   **Llama 2:** Run decode and fine-tuning using [these instructions](getting_started/Run_Llama2.md).
*   **Mixtral:** Run decode and fine-tuning using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:** Run pre-training, fine-tuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Explore the comprehensive end-to-end tests in [end_to_end](end_to_end) and the continuous [unit tests](.github/workflows/RunTests.yml) for more capabilities.

## Runtime Performance Results

Detailed results and configurations can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText distinguishes itself from alternatives like MinGPT/NanoGPT, Megatron-LM, and PaxML through its focus on pure Python/Jax implementation, high scalability, and a user-friendly approach that encourages direct code modification. It provides comparable or superior MFU compared to other frameworks, while maintaining simplicity and ease of use.

## Features and Diagnostics

### Collect Stack Traces

Enable `collect_stack_trace: True` and configure `stack_trace_to_cloud` and `stack_trace_interval_seconds` in `MaxText/configs/base.yml` to debug errors during SPMD jobs. View traces in Cloud Logging using the query:

```
logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
jsonPayload.verb="stacktraceanalyzer"
```

### Ahead of Time Compilation (AOT)

Use the `train_compile.py` tool for AOT compilation to identify potential OOM issues and improve startup times.  Install dependencies and follow the example usages for both TPUs and GPUs to optimize compilation for your specific hardware.  Make sure you match your compilation environment and flags with your execution environment.

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to upload logs to a Vertex AI Tensorboard instance.

### Monitor Goodput of Your Workload

Follow this [user guide](getting_started/Monitor_Goodput.md) for information on monitoring Goodput metrics.

---

**[Explore the MaxText repository on GitHub](https://github.com/AI-Hypercomputer/maxtext) to start building and training your LLMs!**