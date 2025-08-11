# MaxText: High-Performance, Open-Source LLM Training and Inference

**MaxText empowers researchers and developers to train and deploy large language models (LLMs) with exceptional performance using pure Python/JAX.**  ([See the original repository](https://github.com/AI-Hypercomputer/maxtext))

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/RunTests.yml)

## Key Features

*   **TPU and GPU Support:** Train and infer LLMs efficiently on Google Cloud TPUs and GPUs.
*   **Training and Inference:**  Comprehensive support for both LLM training and inference workflows.
*   **Optimized Performance:** Achieve high Model Flops Utilization (MFU) for fast training and inference.
*   **Scalability:** Scale from single-host to very large clusters.
*   **Open Source:**  A flexible, open-source foundation for ambitious LLM projects.
*   **Modular Imports:** Offers API changes for easier development.
*   **Multi-Token Prediction (MTP) Support:** Enhances training efficiency.
*   **Model Compatibility:**  Supports popular LLM families including:
    *   Llama 2, Llama 3, Llama 4 (Maverick, Scout)
    *   Mistral and Mixtral
    *   Gemma, Gemma 2, Gemma 3
    *   DeepSeek family

## Announcements

*   **(July 27, 2025)** Performance updates for attention calculations (see [PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988), [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009), and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030)).
*   **(July 16, 2025)** Repository restructuring planned (see [RESTRUCTURE.md]).
*   **(July 11, 2025)** Multi-Token Prediction (MTP) training support.
*   **(June 25, 2025)** DeepSeek R1-0528 variant support.
*   **(April 24, 2025)** Llama 4 Maverick model support.
*   **(April 14, 2025)** Llama 4 Scout model support.
*   **(April 7, 2025)** Modular import API change for `train.py`.
*   **(April 2, 2025)** DeepSeek v3-0324 variant support.
*   **(March 24, 2025)** DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **(March 12, 2025)** Gemma 3 (4B, 12B, 27B) support.
*   **(February, 2025)** Preview of MaxText Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Jumpstart your journey with MaxText using the [First Run instructions](getting_started/First_run.md). These guides provides information on running decode and finetuning for different families of LLMs:

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

Explore comprehensive end-to-end tests in the [end\_to\_end](end_to_end) folder and continuous [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

Find detailed performance results below:

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

MaxText distinguishes itself from alternatives like MinGPT, Nvidia/Megatron-LM and Pax by offering a pure Python/JAX implementation, enabling users to easily experiment, scale, and optimize LLMs, as well as achieve high performance.

## Features and Diagnostics

*   **Collect Stack Traces:** Use these configurations to help identify and troubleshoot issues for your jobs: `collect_stack_trace`, `stack_trace_to_cloud`, and `stack_trace_interval_seconds`.
*   **Ahead of Time Compilation (AOT):** Compile your training run ahead of time using `train_compile.py` to identify potential OOM issues, and speed up startup.
    *   **TPU Support:** Requires `jax[tpu]`.
    *   **GPU Support:** Provides support for A3 Cloud GPUs.
*   **Automatically Upload Logs to Vertex Tensorboard:** Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to upload logs.
*   **Monitor Goodput of Your Workload:** Learn how to monitor Goodput metrics with this [user guide](getting_started/Monitor_Goodput.md).