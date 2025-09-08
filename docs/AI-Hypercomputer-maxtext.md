# MaxText: Train and Deploy LLMs with High Performance and Scalability

[Link to Original Repo](https://github.com/AI-Hypercomputer/maxtext)

Maximize your LLM potential with MaxText, a cutting-edge, open-source framework built for training and inference on TPUs and GPUs.

**Key Features:**

*   **High Performance:** Achieve exceptional throughput with optimized Python/Jax code.
*   **Scalability:** Scale training from single-host to massive clusters with ease.
*   **Open Source:** Leverage a transparent and adaptable codebase for research and production.
*   **TPU & GPU Support:** Optimized for Google Cloud TPUs and GPUs.
*   **Training & Inference:** Comprehensive support for both model training and deployment.
*   **Model Support:** Llama 2/3/4, Mistral/Mixtral, Gemma, DeepSeek, and Qwen3 families.

## Overview

MaxText is a **high-performance**, **highly scalable**, **open-source** LLM framework written in pure Python/Jax and optimized for Google Cloud TPUs and GPUs for both **training** and **inference**. Designed for ambitious LLM projects, MaxText delivers exceptional [MFUs](#runtime-performance-results) and scales seamlessly from single-host to very large clusters, while maintaining simplicity thanks to Jax and the XLA compiler.

## Announcements

*   [August 13, 2025] Support for the Qwen3 MoE family of models, starting with Qwen3-235B-A22B-Thinking-2507, is now available.
*   [July 27, 2025] Updated TFLOPS/s calculation to account for causal attention and sliding window/chunked attention.
*   [July 16, 2025] Repository restructuring for improved organization. Please review the [proposed structure](RESTRUCTURE.md) and provide feedback.
*   [July 11, 2025] Multi-Token Prediction (MTP) training support, inspired by the [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1).
*   [June 25, 2025] DeepSeek R1-0528 variant support.
*   [April 24, 2025] Llama 4 Maverick models support.
*   [April 14, 2025] Llama 4 Scout models support (text-only, 8k context).
*   **[April 7, 2025] 🚨🚨🚨 Modular imports and API change for `train.py`:  `python3 -m MaxText.train src/MaxText/configs/base.yml run_name=...`**
*   [April 2, 2025] DeepSeek v3-0324 variant support.
*   [March 24, 2025] Support for DeepSeek v3 (671B) and v2-Lite (16B).
*   [March 12, 2025] Support for Gemma 3: 4B, 12B, and 27B.
*   [February, 2025] (Preview):  Building MaxText Docker images using JAX AI Training Images. Learn more [Here](getting_started/Run_MaxText_via_xpk.md)

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get started with MaxText by following the [first-run instructions](getting_started/First_run.md). Explore our guides to learn how to train and infer various models:

*   **Gemma:**  A family of open-weights LLMs by [Google DeepMind](https://deepmind.google/). Run decode and finetuning with [these instructions](end_to_end/tpu/gemma/Run_Gemma.md).  Use the corresponding [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts for checkpoint convertion and decoding of Gemma 2 and 3.
*   **Llama2:**  Meta's open-weights LLMs.  Follow [these instructions](getting_started/Run_Llama2.md) for decode and finetuning.
*   **Mixtral:** Mistral AI's open-weights sparse mixture-of-experts (MoE) models.  Run decode and finetuning using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:**  DeepSeek AI's novel open-weights MoE models. Instructions for pre-training, finetuning, and decoding are available [here](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Explore the complete suite of end-to-end tests within the [end_to_end](end_to_end) folder, and the continuous [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

Reproduce these results using the information in [src/MaxText/configs/README.md](src/MaxText/configs/README.md).

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

For 16B, 32B, 64B, and 128B models. See full run configs in [src/MaxText/configs/v5e/](src/MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText draws inspiration from [MinGPT](https://github.com/karpathy/minGPT) and [NanoGPT](https://github.com/karpathy/nanoGPT), exceeding their capabilities by scaling to tens of thousands of chips and offering an MFU more than three times the reported values. MaxText is similar to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) in terms of MFU, but differs in its pure Python/Jax approach, relying heavily on the XLA compiler. MaxText is also comparable to [Pax](https://github.com/google/paxml) with a focus on simplicity.

## Features and Diagnostics

### Collect Stack Traces

Configure `collect_stack_trace: True`, `stack_trace_to_cloud: True`, and `stack_trace_interval_seconds` in `src/MaxText/configs/base.yml` to aid debugging SPMD jobs.  View traces in Cloud Logging using the query:
```
logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
jsonPayload.verb="stacktraceanalyzer"
```
More information can be found in the following PyPI package https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile the `train_step` for target hardware, improving startup and restart times.

#### TPU Support

*   Install `jax[tpu]` and other dependencies using `setup.sh`.
*   Compile using a CPU or single VM (e.g., for v5e-256):
    ```bash
    export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
    python3 -m MaxText.train_compile src/MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 global_parameter_scale=16 per_device_batch_size=4
    ```
*   Load the compiled function:
    ```bash
    export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
    python3 -m MaxText.train src/MaxText/configs/base.yml run_name=example_load_compile compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 per_device_batch_size=4 steps=10000 learning_rate=1e-3 base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
    ```

#### GPU Support

*   GPU does not support compilation across hardware.
*   For A3 Cloud GPUs, `compile_topology_num_slices` represents the number of A3 machines.
*   Example (multihost compilation):
    ```bash
    # Compile
    export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
    python3 -m MaxText.train_compile src/MaxText/configs/base.yml compile_topology=a3 compile_topology_num_slices=4 compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3

    # Run on each host
    export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
    python3 -m MaxText.train src/MaxText/configs/base.yml run_name=example_load_compile compiled_trainstep_file=my_compiled_train.pickle attention=dot_product global_parameter_scale=16 per_device_batch_size=4 steps=10000 learning_rate=1e-3 base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
    ```

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to automatically upload logs to Vertex AI Tensorboard.

### Monitor Goodput of Your Workload

Monitor Goodput metrics using the [user guide](getting_started/Monitor_Goodput.md).