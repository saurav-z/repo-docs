# MaxText: High-Performance LLM Training and Inference in Pure Python/JAX

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/RunTests.yml)

**MaxText empowers researchers and engineers to train and deploy large language models (LLMs) at scale with exceptional performance using Python and JAX.**  This open-source project is designed for both research and production, offering a streamlined, optimization-free approach to LLM development.  [View the original repository](https://github.com/AI-Hypercomputer/maxtext).

## Key Features

*   **High Performance:** Achieves industry-leading Model Flops Utilization (MFU) through optimized JAX/XLA compilation.
*   **Scalability:** Designed to scale seamlessly from single GPUs/TPUs to massive clusters.
*   **Open Source:** Fully open-source and written in pure Python/JAX.
*   **Training and Inference:** Supports both training and inference workloads.
*   **Hardware Support:** Optimized for Google Cloud TPUs and GPUs.
*   **Model Support:** Compatible with a wide array of open-source models, including:
    *   Llama 2, Llama 3, Llama 4
    *   Mistral and Mixtral family
    *   Gemma, Gemma 2, Gemma 3
    *   DeepSeek
    *   Qwen3 Dense and MoE family
*   **Modular Imports:** Enables streamlined development workflow via modular imports.

## Recent Announcements

*   **[August 13, 2025]** Added support for the Qwen3 MoE family of models.
*   **[July 27, 2025]** Updated TFLOPS/s calculation for causal, sliding window and chunked attention
*   **[July 16, 2025]** Repository restructuring for improved clarity.
*   **[July 11, 2025]** Introduced Multi-Token Prediction (MTP) training support, inspired by the DeepSeek-V3 paper.
*   **[June 25, 2025]** Added DeepSeek R1-0528 support.
*   **[April 24, 2025]** Llama 4 Maverick model support.
*   **[April 14, 2025]** Llama 4 Scout model support.
*   **[April 7, 2025]** Modular import API change for train.py.
*   **[April 2, 2025]** Added DeepSeek v3-0324 support.
*   **[March 24, 2025]** Added DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]** Added Gemma 3 support.
*   **[February, 2025]** (Preview) Preview of building Maxtext Docker images using the JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Jumpstart your LLM journey with our comprehensive [Getting Started](getting_started/First_run.md) guide.  MaxText is designed for both novice and experienced users.

Explore the following resources for various models:

*   **Gemma:** Run decode and finetuning using the [Gemma guide](end_to_end/tpu/gemma/Run_Gemma.md). (Also Gemma 2/3 via scripts in corresponding folders.)
*   **Llama2:** Follow these [instructions](getting_started/Run_Llama2.md) to run decode and finetuning.
*   **Mixtral:** Check out the [Mixtral guide](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:** Learn how to pre-train, finetune, and decode with our [DeepSeek guide](end_to_end/tpu/deepseek/Run_DeepSeek.md).

For ongoing updates, explore the suite of end-to-end tests in [end_to_end](end_to_end), updated nightly, or the continuous [unit tests](.github/workflows/RunTests.yml).

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

MaxText offers a compelling alternative to other LLM implementations, particularly when considering its Python/JAX foundation and focus on performance and scalability. Compared to MinGPT/NanoGPT, MaxText provides superior MFU, scales to massive clusters, and supports key-value caching for efficient auto-regressive decoding.  While similar to Megatron-LM in achieving high MFU, MaxText differentiates through its pure Python/JAX approach, leveraging the XLA compiler for optimization.  In comparison to Pax, MaxText prioritizes a straightforward, concrete implementation, making it an ideal starting point for users interested in forking and modifying the source code.

## Features and Diagnostics

### Collect Stack Traces

Enable `collect_stack_trace: True` in `MaxText/configs/base.yml` to gather stack traces for troubleshooting errors. Use `stack_trace_to_cloud: True` to upload traces to Cloud Logging.

### Ahead of Time Compilation (AOT)

Compile your training runs ahead of time for faster startup times and to identify OOM errors early.  The `train_compile.py` tool allows you to precompile the `train_step` for specific hardware configurations.

*   **TPU Support:**  Compile for TPUs using a single CPU or VM; see examples.
*   **GPU Support:**  Compile for GPUs using a single GPU host; see examples.

### Automatically Upload Logs to Vertex Tensorboard

Integrate with Vertex AI TensorBoard to automatically upload logs.  Refer to the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for more information.

### Monitor Goodput of Your Workload

Monitor Goodput metrics; please refer to this [user guide](getting_started/Monitor_Goodput.md).