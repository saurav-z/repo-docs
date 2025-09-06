# MaxText: High-Performance LLM Training and Inference with Jax

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxtext)

**MaxText is a cutting-edge, open-source LLM framework, empowering you to train and run large language models efficiently on TPUs and GPUs.** Built in pure Python/Jax, MaxText delivers exceptional performance and scalability for both training and inference. Explore the MaxText [original repository](https://github.com/AI-Hypercomputer/maxtext) for more details!

## Key Features

*   **TPU & GPU Support:** Optimized for Google Cloud TPUs and GPUs.
*   **Training and Inference:** Comprehensive support for both model training and deployment.
*   **High Performance:** Achieves impressive Model Flops Utilization (MFU) thanks to Jax and XLA.
*   **Scalability:** Scales from single devices to massive clusters.
*   **Open Source & Customizable:**  Start experimenting with MaxText and easily modify it to meet your specific needs.
*   **Model Support:** Ready to use with a variety of models including Llama 2, Llama 3, Mistral, Mixtral, Gemma, DeepSeek, and Qwen3 families.

## Recent Updates & Announcements

*   **[August 13, 2025]** Qwen3 MoE family support (Qwen3-235B-A22B-Thinking-2507).
*   **[July 27, 2025]** Updated TFLOPS/s calculations accounting for causal, sliding window, and chunked attention.
*   **[July 16, 2025]** Repository restructuring for improved organization.  See [RESTRUCTURE.md](RESTRUCTURE.md).
*   **[July 11, 2025]** Multi-Token Prediction (MTP) training support.
*   **[June 25, 2025]** DeepSeek R1-0528 variant support.
*   **[April 24, 2025]** Llama 4 Maverick model support.
*   **[April 14, 2025]** Llama 4 Scout model support (text-only, 8k context).
*   **[April 7, 2025]** Modular imports are now supported.  API change in `train.py`.
*   **[April 2, 2025]** DeepSeek v3-0324 variant support.
*   **[March 24, 2025]** DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]** Gemma 3: 4B, 12B, and 27B support.
*   **[February, 2025]** (Preview): Maxtext Docker image building via JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get up and running with MaxText quickly with our [detailed instructions](getting_started/First_run.md).

Explore our guides for training and inference on various open models within the [getting started](getting_started) directory.

Helpful guides for specific models:

*   **Gemma:** [End-to-end instructions](end_to_end/tpu/gemma/Run_Gemma.md) for a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini research and technology. For Gemma 2 and 3, use the corresponding [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts for checkpoint conversion and decoding.
*   **Llama2:** [Instructions](getting_started/Run_Llama2.md) for Meta's family of open-weights Large Language Models (LLM).
*   **Mixtral:** [Instructions](end_to_end/tpu/mixtral/Run_Mixtral.md) for a family of open-weights sparse mixture-of-experts (MoE) models by Mistral AI.
*   **DeepSeek:**  [Instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md) for a novel family of open-weights sparse MoE models by DeepSeek AI, featuring advanced techniques for enhanced efficiency and performance.

Also see our end-to-end tests in [end_to_end](end_to_end) for understanding the capabilities of MaxText.  Unit tests can be found [here](.github/workflows/RunTests.yml).

## Runtime Performance Results

Detailed results are in [src/MaxText/configs/README.md](src/MaxText/configs/README.md).

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

See configs in [src/MaxText/configs/v5e/](src/MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText is a high-performance alternative to existing LLM frameworks. It offers:

*   **Superior Performance:** Achieves significantly higher MFU than implementations like MinGPT and NanoGPT.
*   **Scalability:** Unlike some alternatives, MaxText is designed to scale to very large clusters.
*   **Codebase Differences**:  MaxText is a pure Python implementation, which makes it distinct from others like Megatron-LM that are a mix of Python and CUDA.  MaxText relies heavily on the XLA compiler to achieve high performance, whereas Megatron-LM relies on well-optimized CUDA kernels to achieve high performance.
*   **Emphasis on Customization:** MaxText encourages direct modification of the code for customized LLM development.

## Features and Diagnostics

### Collect Stack Traces

Enable stack trace collection for debugging SPMD jobs using the following configurations in `src/MaxText/configs/base.yml`:

1.  Set `collect_stack_trace: True` to enable. Set to `False` to disable.
2.  Set `stack_trace_to_cloud: False` to display traces on console. `stack_trace_to_cloud: True` to store traces in `/tmp/debugging`. View traces in Logs Explorer using the query provided.
3.  `stack_trace_interval_seconds` controls the collection interval.

See https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` for AOT compilation for faster startup times. Requires `jax[tpu]` installation.

*   **TPU Support:** Pre-compile for TPUs to identify OOM issues and save compiled functions.
    *   See example usage with `compile_topology`,  `compiled_trainstep_file`, and  `global_parameter_scale`.
*   **GPU Support:** Similar AOT support for GPUs.
    *   See example using `XLA_FLAGS`.

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to set up automatic log uploads to a Vertex AI TensorBoard instance.

### Monitor Goodput of Your Workload

Monitor Goodput metrics using the guide [here](getting_started/Monitor_Goodput.md).