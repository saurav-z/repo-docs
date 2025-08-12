# MaxText: High-Performance, Open-Source LLM Training and Inference with Jax

**Maximize your LLM performance with MaxText, a powerful, open-source framework built in pure Python/Jax for efficient training and inference on TPUs and GPUs.** ([View on GitHub](https://github.com/AI-Hypercomputer/maxtext))

MaxText empowers researchers and production teams to build and deploy large language models with exceptional speed and scalability.

**Key Features:**

*   **High Performance:** Achieves exceptional Model Flops Utilization (MFU) for rapid training and inference.
*   **Scalable:** Designed to scale from single-host setups to massive clusters, supporting models with trillions of parameters.
*   **Open Source:** Built on pure Python/Jax, fostering community collaboration and customization.
*   **TPU and GPU Support:** Optimized for Google Cloud TPUs and GPUs.
*   **Training and Inference:** Versatile framework for both model training and deployment.
*   **Model Support:** Compatible with a wide range of models, including:
    *   Llama 2, Llama 3, Llama 4
    *   Mistral and Mixtral family
    *   Gemma, Gemma 2, Gemma 3
    *   DeepSeek family

## Announcements

*   [July 27, 2025] Updated TFLOPS/s calculation to account for causal attention and sliding window/chunked attention.
*   [July 16, 2025] Repository restructuring for improved organization and clarity ([RESTRUCTURE.md](RESTRUCTURE.md) for proposed structure).
*   [July 11, 2025] Added Multi-Token Prediction (MTP) training support.
*   [June 25, 2025] DeepSeek R1-0528 variant support.
*   [April 24, 2025] Llama 4 Maverick models support.
*   [April 14, 2025] Llama 4 Scout models support (8k context length).
*   **[April 7, 2025] API change: Modular imports, use `python3 -m MaxText.train ...` instead of direct execution. For older API, use `git checkout pre-module-v0.1.0`.**
*   [April 2, 2025] DeepSeek v3-0324 variant support.
*   [March 24, 2025] DeepSeek v3 (671B) and v2-Lite (16B) support.
*   [March 12, 2025] Gemma 3 (4B, 12B, 27B) support.
*   [February, 2025] (Preview) MaxText Docker image builds using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get started quickly with MaxText using our detailed [instructions](getting_started/First_run.md).  Explore our guides for running and fine-tuning popular open models:

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

Explore the [end_to_end](end_to_end) directory for a comprehensive suite of end-to-end tests and see the continuous [unit tests](.github/workflows/RunTests.yml) for understanding MaxText capabilities.

## Runtime Performance Results

Reproduce these results using configurations in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText offers a compelling alternative to existing LLM implementations, providing superior performance and flexibility:

*   **Compared to MinGPT/NanoGPT:** MaxText provides higher MFU, is scalable, and implements a key-value cache for efficient decoding.
*   **Compared to Megatron-LM:** MaxText offers comparable MFU, while differentiating with a pure Python/Jax codebase.
*   **Compared to Pax:** MaxText offers a simpler, more concrete approach, encouraging direct code modification rather than configuration-driven changes.

## Features and Diagnostics

### Collect Stack Traces

Configure stack trace collection to diagnose issues in SPMD jobs:
1.  Set `collect_stack_trace: True` to enable trace collection on faults.
2.  `stack_trace_to_cloud: True` to store traces in GCP Cloud Logging (query: `logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor" jsonPayload.verb="stacktraceanalyzer"`). Use `stack_trace_to_cloud: False` for console display.
3.  `stack_trace_interval_seconds` sets the collection interval.

Related PyPI package: [cloud-tpu-diagnostics](https://pypi.org/project/cloud-tpu-diagnostics).

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile `train_step` for faster startup and OOM detection:

*   **TPU Support:** Compile using CPU or single VM.
*   **GPU Support:**  Compile for multihost GPU clusters; A single GPU host can compile for the same hardware.
*   See Examples in original README.

### Automatically Upload Logs to Vertex Tensorboard

Integrate with Vertex AI Tensorboard for automatic log uploads.  See the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

Monitor Goodput metrics. See the [user guide](getting_started/Monitor_Goodput.md).