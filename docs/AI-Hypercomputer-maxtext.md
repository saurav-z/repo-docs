# MaxText: High-Performance, Scalable LLM Training and Inference 🚀

**Maximize your LLM performance with MaxText, a pure Python/Jax framework designed for training and inference on Google Cloud TPUs and GPUs, achieving high MFUs and scaling to massive clusters.** ([View on GitHub](https://github.com/AI-Hypercomputer/maxtext))

## Key Features:

*   **High Performance:** Optimized for speed and efficiency using Jax and the XLA compiler, achieving industry-leading Model Flops Utilization (MFU).
*   **Scalability:** Designed to scale from single-host to extremely large clusters, enabling training of the largest LLMs.
*   **Open Source:** Built on open-source principles, fostering collaboration and customization.
*   **TPU and GPU Support:** Compatible with Google Cloud TPUs and GPUs, offering flexibility in hardware choices.
*   **Training and Inference:** Comprehensive support for both training and inference workflows.
*   **Model Support:** Includes a wide range of popular LLMs: Llama 2, Llama 3, Llama 4, Mistral and Mixtral family, Gemma, Gemma 2, Gemma 3, DeepSeek, Qwen3 Dense and MoE family.

## Announcements

*   **[August 13, 2025]:** Added support for the Qwen3 MoE family of models.
*   **[July 27, 2025]:** Updated TFLOPS/s calculation to account for causal attention and chunked attention.
*   **[July 16, 2025]:** Restructuring of the MaxText repository is proposed.
*   **[July 11, 2025]:** Added Multi-Token Prediction (MTP) training.
*   **[June 25, 2025]:** DeepSeek R1-0528 variant is now supported!
*   **[April 24, 2025]:** Llama 4 Maverick models are now supported!
*   **[April 14, 2025]:** Llama 4 Scout models are now supported.
*   **[April 7, 2025]:** Modular Imports supported with an API change for `train.py`.
*   **[April 2, 2025]:** DeepSeek v3-0324 variant is now supported!
*   **[March 24, 2025]:** Added support for DeepSeek v3 (671B) and v2-Lite (16B).
*   **[March 12, 2025]:** Added support for Gemma 3: 4B, 12B, and 27B.
*   **[February, 2025]:** (Preview): Building Maxtext Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Quickly get up and running with MaxText using our detailed [instructions](getting_started/First_run.md).

MaxText provides comprehensive support for training and inference of various open models. Find the user guides in the [getting started](getting_started) folder.

**Explore the supported models:**

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma): a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini research and technology. You can run decode and finetuning using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md). For Gemma 2 and 3, use the corresponding [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts for checkpoint convertion and decoding.
*   [Llama2](https://llama.meta.com/llama2/): a family of open-weights Large Language Model (LLM) by Meta. You can run decode and finetuning using [these instructions](getting_started/Run_Llama2.md).
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): a family of open-weights sparse mixture-of-experts (MoE) models by Mistral AI. You can run decode and finetuning using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): a novel family of open-weights sparse MoE models by DeepSeek AI. DeepSeek-V3 features advanced techniques, including Multi-Head Latent Attention (MLA), finer-grained and shared experts, Multi-Token Prediction (MTP), and FP8 mixed precision designed for enhanced efficiency and performance. You can run pre-training, finetuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Access our continuously updated end-to-end tests in [end_to_end](end_to_end), or refer to the continuous [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

See the following performance metrics for supported hardware. More details on reproducing these results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

## TPU v5p

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

## TPU v5e

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

MaxText offers a high-performance alternative to [MinGPT](https://github.com/karpathy/minGPT)/[NanoGPT](https://github.com/karpathy/nanoGPT) and [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), achieving comparable or superior MFUs, especially scaling to large LLM training jobs. While MaxText is pure Python, it leverages the power of the XLA compiler and is simple and concrete.

## Features and Diagnostics

### Collect Stack Traces

Configure `collect_stack_trace: True`,  `stack_trace_to_cloud: True`, and  `stack_trace_interval_seconds` in `MaxText/configs/base.yml` to debug SPMD job issues, and review traces in Cloud Logging using the query:
```
logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
jsonPayload.verb="stacktraceanalyzer"
```

### Ahead of Time Compilation (AOT)

The `train_compile.py` tool pre-compiles `train_step` for your target hardware, helping with OOM detection and fast startup/restart. For TPU and GPU support, follow the examples to save then load compiled functions. The environment used in compilation must match the execution environment, e.g. by setting the same `XLA_FLAGS`.

### Automatically Upload Logs to Vertex Tensorboard

Upload logs automatically using Vertex AI. See [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for details.

### Monitor Goodput of Your Workload

Use our [user guide](getting_started/Monitor_Goodput.md) to monitor your workload's Goodput metrics.