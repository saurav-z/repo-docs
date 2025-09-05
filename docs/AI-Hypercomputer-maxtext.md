# MaxText: High-Performance, Scalable LLM Training and Inference

**Maximize your LLM potential with MaxText, a high-performance, open-source framework built for training and inference of large language models using pure Python/JAX.** ([Original Repo](https://github.com/AI-Hypercomputer/maxtext))

## Key Features

*   **High Performance:** Achieves high Model Flops Utilization (MFU) for efficient training and inference.
*   **Scalable:**  Designed to scale from single-host to massive clusters, supporting very large language models.
*   **Open Source:**  Leverage the power of open-source for flexibility and customization.
*   **TPU and GPU Support:** Optimized for Google Cloud TPUs and GPUs, enabling efficient use of hardware resources.
*   **Training and Inference:** Supports both model training and inference tasks.
*   **Model Support:**  Supports a wide range of models including Llama 2, Llama 3, Mistral, Mixtral, Gemma, DeepSeek, and Qwen3 families.

## Announcements

*   **[August 13, 2025]** Qwen3 MoE family of models is now supported.
*   **[July 27, 2025]** Updated TFLOPS/s calculation to account for causal attention.
*   **[July 16, 2025]** Repository restructuring proposed for improved organization.
*   **[July 11, 2025]** Multi-Token Prediction (MTP) training is now supported!
*   **[June 25, 2025]** DeepSeek R1-0528 variant is now supported!
*   **[April 24, 2025]** Llama 4 Maverick models are now supported!
*   **[April 14, 2025]** Llama 4 Scout models are now supported.
*   **[April 7, 2025]** Modular imports are now supported with API changes.
*   **[April 2, 2025]** DeepSeek v3-0324 variant is now supported!
*   **[March 24, 2025]** Support for DeepSeek v3 (671B) and v2-Lite (16B) is now available.
*   **[March 12, 2025]** Support for Gemma 3: 4B, 12B, and 27B is now available.
*   **[February, 2025]** (Preview) Maxtext Docker images using the JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get started with MaxText quickly with our comprehensive [Getting Started](getting_started/First_run.md) guide.  Explore the [getting started](getting_started) folder for detailed instructions on training and inference for various open models.

Explore the following guides:
*   [Gemma (generations 1-3)](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

The full suite of end-to-end tests is available in [end_to_end](end_to_end).

## Runtime Performance Results

MaxText delivers impressive performance on TPUs and GPUs.  See below for benchmark results. More details on reproducing these results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText offers significant performance advantages compared to other LLM implementations.

*   **Compared to MinGPT/NanoGPT:**  MaxText provides a significantly higher MFU and is designed for much larger scale.
*   **Compared to Megatron-LM:**  MaxText achieves comparable MFUs with a pure Python/JAX implementation, highlighting different programming strategies.
*   **Compared to Pax:** MaxText offers a simpler, more accessible codebase encouraging users to customize by modifying the code.

## Features and Diagnostics

### Collect Stack Traces

Enable stack trace collection to aid in debugging SPMD jobs.  Configure the following parameters in `MaxText/configs/base.yml`:

1.  Set `collect_stack_trace: True` to enable stack trace collection.
2.  Set `stack_trace_to_cloud: True` to upload traces to Cloud Logging, or `stack_trace_to_cloud: False` to display them on the console.  View traces in Cloud Logging using the provided query.
3.  Adjust `stack_trace_interval_seconds` to control the frequency of stack trace collection.

### Ahead of Time Compilation (AOT)

Compile your training runs ahead of time using `train_compile.py` for faster startup and restart times.
*   **TPU Support:**  Use a CPU or a single VM to pre-compile for a TPU cluster. AOT compilation can catch OOM errors.  Save and load the compiled function.
*   **GPU Support:**  A single GPU host can compile for a larger GPU cluster of the same hardware.

    Example usages are given in the provided documentation.

### Automatically Upload Logs to Vertex Tensorboard

Use MaxText to automatically upload logs to a Vertex AI Tensorboard instance.  Refer to the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for more information.

### Monitor Goodput of Your Workload

Monitor the goodput metrics of your workload using the guide provided in [getting_started/Monitor_Goodput.md](getting_started/Monitor_Goodput.md).