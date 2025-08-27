# MaxText: High-Performance, Open-Source LLM Training and Inference

**MaxText empowers you to train and deploy large language models (LLMs) with exceptional speed and scalability using pure Python/Jax.** ([Original Repo](https://github.com/AI-Hypercomputer/maxtext))

## Key Features

*   **High Performance:** Achieves high Model Flops Utilization (MFU) for efficient training and inference.
*   **Scalability:** Designed to scale from single hosts to massive clusters on Google Cloud TPUs and GPUs.
*   **Open Source:**  Freely available and customizable for research and production.
*   **TPU and GPU Support:**  Optimized for Google Cloud TPUs (v5e, v5p) and GPUs (A3).
*   **Training and Inference:** Supports both model training and inference tasks.
*   **Model Compatibility:** Supports a wide range of models, including:
    *   Llama 2, Llama 3, Llama 4
    *   Mistral and Mixtral family
    *   Gemma, Gemma 2, Gemma 3
    *   DeepSeek
    *   Qwen3 Dense and MoE family

## What's New

*   **[August 13, 2025]** Added support for the Qwen3 MoE family of models, starting with Qwen3-235B-A22B-Thinking-2507, Qwen3-30B-A3B and Qwen3-Coder-480B-A35B.
*   **[July 27, 2025]**  Updated TFLOPS/s calculation to account for causal, sliding window, and chunked attention, improving accuracy.
*   **[July 16, 2025]**  Restructuring the repository for improved clarity.
*   **[July 11, 2025]**  Added Multi-Token Prediction (MTP) training support.
*   **[June 25, 2025]**  Added DeepSeek R1-0528 variant support.
*   **[April 24, 2025]** Added Llama 4 Maverick model support.
*   **[April 14, 2025]** Added Llama 4 Scout model support (text-only, 8k context, optimizations in progress).
*   **[April 7, 2025]**  Modular imports support with an API change for train.py.
*   **[April 2, 2025]**  Added DeepSeek v3-0324 variant support.
*   **[March 24, 2025]**  Added DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]**  Added Gemma 3 (4B, 12B, 27B) support.
*   **[February, 2025]** Preview of building Maxtext Docker images using the JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Dive into LLM training and inference with MaxText using our comprehensive guides:
*   [First Run](getting_started/First_run.md)

Explore specific model implementations:
*   [Gemma](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

## Runtime Performance Results

Achieve optimal performance with MaxText.  Find details in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText builds upon concepts from [MinGPT](https://github.com/karpathy/minGPT)/[NanoGPT](https://github.com/karpathy/nanoGPT) and achieves significantly higher MFU.  It's similar to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), but employs a pure Python/JAX approach.  Compared to [Pax](https://github.com/google/paxml), MaxText prioritizes a direct, code-centric approach for customization.

## Features and Diagnostics

*   **Collect Stack Traces:**  Improve debugging on SPMD jobs.  Enable with `collect_stack_trace: True` in `configs/base.yml`. Configure output with `stack_trace_to_cloud`.  View traces in Logs Explorer with the given query.
*   **Ahead of Time Compilation (AOT):** Optimize training startup and reduce memory errors.
    *   **TPU Support:** Uses a CPU or single VM to pre-compile for a TPU cluster. See examples.
    *   **GPU Support:**  Pre-compilation for GPU clusters, with examples provided.  Use `XLA_FLAGS="--xla_gpu_enable_async_collectives=true"`
*   **Automatically Upload Logs to Vertex Tensorboard:**  Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to get started.
*   **Monitor Goodput of Your Workload:** Refer to the [user guide](getting_started/Monitor_Goodput.md) to learn more.