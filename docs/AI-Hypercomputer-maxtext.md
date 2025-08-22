# MaxText: High-Performance, Open-Source LLM Training & Inference 🚀

**MaxText empowers you to train and run large language models (LLMs) with exceptional performance and scalability using pure Python/JAX.** ([Original Repository](https://github.com/AI-Hypercomputer/maxtext))

## Key Features:

*   **High Performance:** Achieves industry-leading Model Flops Utilization (MFU) for efficient training and inference.
*   **Scalable:** Designed to scale from single hosts to massive clusters on Google Cloud TPUs and GPUs.
*   **Open Source:**  Freely available and customizable, enabling experimentation and adaptation to your specific needs.
*   **TPU & GPU Support:** Optimized for Google Cloud TPUs (v5e, v5p) and GPUs.
*   **Training & Inference:** Supports both training and inference workflows.
*   **Model Support:** Ready-to-use with popular models including Llama 2, Llama 3, Gemma, DeepSeek, Mixtral, and Qwen3 families.

## What's New:

*   **[August 13, 2025]** Added support for the Qwen3 MoE family of models.
*   **[July 27, 2025]** Updated TFLOPS/s calculations to account for causal, sliding window, and chunked attention.
*   **[July 16, 2025]** Restructuring the MaxText repository.
*   **[July 11, 2025]** Added support for Multi-Token Prediction (MTP) training.
*   **[June 25, 2025]** Added support for the DeepSeek R1-0528 variant.
*   **[April 24, 2025]** Added support for Llama 4 Maverick models.
*   **[April 14, 2025]** Added support for Llama 4 Scout models.
*   **[April 7, 2025]** Modular imports: API change for `train.py`.
*   **[April 2, 2025]** Added support for the DeepSeek v3-0324 variant.
*   **[March 24, 2025]** Added support for DeepSeek v3 (671B) and v2-Lite (16B).
*   **[March 12, 2025]** Added support for Gemma 3: 4B, 12B, and 27B.
*   **[February, 2025]** (Preview): Building Maxtext Docker images using the JAX AI Training Images.

## Table of Contents:

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started:

Kickstart your journey with MaxText using the [Getting Started](getting_started/First_run.md) instructions.

Explore support for various open models:

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma):  Run decode and finetuning using these instructions.
*   [Llama2](https://llama.meta.com/llama2/): Run decode and finetuning using these instructions.
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Run decode and finetuning using these instructions.
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Run pre-training, finetuning, and decoding using these instructions.

## Runtime Performance Results

MaxText delivers exceptional performance. Find details on reproducing these results in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText is inspired by and improves upon existing LLM implementations.  It offers:

*   **Higher MFU:**  Exceeds performance of similar projects.
*   **Scalability:** Designed to operate in large clusters.
*   **Programming Strategies:** Utilizes pure Python/JAX, relying on the XLA compiler for high performance.

## Features and Diagnostics

### Collect Stack Traces
Configure `collect_stack_trace`, `stack_trace_to_cloud` and `stack_trace_interval_seconds` in `MaxText/configs/base.yml` to debug issues by capturing and storing stack traces.

### Ahead of Time Compilation (AOT)
Compile your training runs ahead of time with the `train_compile.py` tool for TPUs and GPUs.  This can help with:

*   Identifying OOM issues.
*   Faster startup and restart times.

#### TPU Support
Follow examples to compile and run with TPU hardware.

#### GPU Support
Follow examples to compile and run with GPU hardware.

### Automatically Upload Logs to Vertex Tensorboard
Follow user guide to automatically upload logs to a Tensorboard instance in Vertex AI.

### Monitor Goodput of Your Workload
Follow the user guide to monitor the goodput metrics of your workload.