# MaxText: High-Performance, Open-Source LLM Training and Inference

**Supercharge your LLM projects with MaxText, a blazing-fast, scalable, and open-source LLM framework built on pure Python/Jax.** [Visit the original repo](https://github.com/AI-Hypercomputer/maxtext)

## Key Features

*   **High Performance:** Achieves exceptional Model Flops Utilization (MFU) for rapid training and inference.
*   **Scalability:** Designed to scale seamlessly from single hosts to massive clusters using Google Cloud TPUs and GPUs.
*   **Open Source:**  Leverage a pure Python/Jax implementation for flexibility and customization.
*   **Training & Inference:** Supports both training and inference workloads.
*   **Model Support:**  Includes support for Llama 2, Llama 3, Mistral, Mixtral, DeepSeek, Qwen3, and Gemma families of models.
*   **TPU and GPU Compatibility:**  Runs efficiently on both Google Cloud TPUs and GPUs.

## Announcements

*   **[August 13, 2025]**: Qwen3 MoE family of models (Qwen3-235B-A22B-Thinking-2507, Qwen3-30B-A3B, and Qwen3-Coder-480B-A35B) are now supported.  Plus existing Qwen3 Dense models.
*   **[July 27, 2025]**:  Attention flop calculations updated for causal, sliding window, and chunked attention, impacting large sequence configurations.
*   **[July 16, 2025]**:  Repository restructuring in progress; review [RESTRUCTURE.md](RESTRUCTURE.md) and provide feedback.
*   **[July 11, 2025]**:  Multi-Token Prediction (MTP) training support added for enhanced training efficiency.
*   **[June 25, 2025]**:  DeepSeek R1-0528 variant supported.
*   **[April 24, 2025]**:  Llama 4 Maverick models supported.
*   **[April 14, 2025]**:  Llama 4 Scout models supported.
*   **[April 7, 2025]**: Modular imports are now supported, with an API change for `train.py`.
*   **[April 2, 2025]**: DeepSeek v3-0324 variant supported.
*   **[March 24, 2025]**:  DeepSeek v3 (671B) and v2-Lite (16B) support added.
*   **[March 12, 2025]**:  Gemma 3 (4B, 12B, and 27B) in text-only format support.
*   **[February, 2025] (Preview)**: MaxText Docker image building using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Begin your MaxText journey with the [First Run instructions](getting_started/First_run.md). Explore guides for specific models:

*   **Gemma:**  Follow instructions for running decode and finetuning [here](end_to_end/tpu/gemma/Run_Gemma.md)
*   **Llama2:**  Instructions for running decode and finetuning [here](getting_started/Run_Llama2.md).
*   **Mixtral:**  Run decode and finetuning using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:**  Instructions for pre-training, finetuning, and decoding [here](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Discover additional capabilities in the [end_to_end](end_to_end) tests and the continuous [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

[MaxText/configs/README.md](MaxText/configs/README.md) contains details on reproducing these results.

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

Inspired by [MinGPT](https://github.com/karpathy/minGPT)/[NanoGPT](https://github.com/karpathy/nanoGPT), MaxText provides a more complex, scalable implementation. MaxText is comparable to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), achieving similar MFU. MaxText is also comparable to [Pax](https://github.com/google/paxml) by providing high-performance and scalable implementations of LLMs in Jax.

## Features and Diagnostics

### Collect Stack Traces

Configure `collect_stack_trace`, `stack_trace_to_cloud`, and `stack_trace_interval_seconds` in `MaxText/configs/base.yml` to debug SPMD jobs.  See the example query in the documentation to view traces in Cloud Logging.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile the `train_step` for faster startup and to catch out-of-memory errors.

*   **TPU Support:** Compile on a CPU or single VM.
*   **GPU Support:**  Compile on a single GPU host for a larger cluster.

See the examples in the original documentation.

### Automatically Upload Logs to Vertex Tensorboard

Automatically upload logs to a Vertex AI Tensorboard instance.  See [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

Monitor Goodput metrics by following this [user guide](getting_started/Monitor_Goodput.md).