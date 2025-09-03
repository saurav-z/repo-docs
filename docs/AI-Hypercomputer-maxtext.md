# MaxText: High-Performance, Scalable LLM Training & Inference

**Maximize your LLM potential with MaxText, a high-performance, open-source solution built in pure Python/Jax for training and inference on Google Cloud TPUs and GPUs.**

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/RunTests.yml)

## Overview

MaxText is a cutting-edge, open-source LLM designed for both research and production. Built with pure Python/Jax, it leverages the power of the XLA compiler to achieve exceptional performance and scalability on Google Cloud TPUs and GPUs. From single-host setups to massive clusters, MaxText delivers high Model Flops Utilization (MFU) and offers a streamlined, "optimization-free" approach.  Experiment, adapt, and build upon MaxText to accelerate your LLM projects.

Key Features:

*   **TPU & GPU Support:** Train and infer on Google Cloud TPUs (v5p, v5e) and GPUs (A3).
*   **Training & Inference:** Comprehensive support for both training and inference workloads.
*   **Model Compatibility:** Supports a wide range of models, including Llama 2/3/4, Mistral/Mixtral, Gemma, DeepSeek, and Qwen3 families.
*   **High Performance:** Achieves industry-leading MFU for efficient resource utilization.
*   **Open Source & Flexible:**  Leverage the power of Jax and XLA for customization.

## Key Announcements

*   **(August 13, 2025)** Added support for the Qwen3 MoE family of models including Qwen3-235B-A22B-Thinking-2507, Qwen3-30B-A3B and Qwen3-Coder-480B-A35B.
*   **(July 27, 2025)**  Updated TFLOPS/s calculation to account for causal attention.
*   **(July 16, 2025)** Restructuring the MaxText repository for improved organization and clarity.
*   **(July 11, 2025)** Multi-Token Prediction (MTP) training support added.
*   **(June 25, 2025)** DeepSeek R1-0528 variant support added.
*   **(April 24, 2025)** Llama 4 Maverick models are now supported!
*   **(April 14, 2025)** Llama 4 Scout models are now supported.
*   **(April 7, 2025)** Modular imports enabled with API change for `train.py`.
*   **(April 2, 2025)** DeepSeek v3-0324 variant support added.
*   **(March 24, 2025)** Support for DeepSeek v3 (671B) and v2-Lite (16B) announced.
*   **(March 12, 2025)** Support for Gemma 3 models announced.
*   **(February, 2025) (Preview):** Docker image builds using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get up and running quickly with MaxText using our comprehensive [instructions](getting_started/First_run.md).

### Model Specific Guides
*   [Gemma (generations 1-3)](https://ai.google.dev/gemma): Open-weights LLM by Google DeepMind.
*   [Llama2](https://llama.meta.com/llama2/): Open-weights LLM by Meta.
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Mixture-of-Experts (MoE) models by Mistral AI.
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Open-weights sparse MoE models by DeepSeek AI.

Explore additional guides in the [getting started](getting_started) directory.  Also check out the [end-to-end](end_to_end) tests for more in-depth usage examples and the continuous [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

Reproduce these results using instructions from [MaxText/configs/README.md](MaxText/configs/README.md).

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

See configurations in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText offers a distinct approach, emphasizing simplicity and high performance through pure Python/Jax, compared to other LLM implementations like Megatron-LM and Pax. MaxText achieves comparable MFUs to Megatron-LM.

## Features and Diagnostics

### Collect Stack Traces

*   **Enable:**  Set `collect_stack_trace: True` in `MaxText/configs/base.yml`.
*   **Cloud Logging:**  Enable `stack_trace_to_cloud: True` to upload traces to Cloud Logging. View in Logs Explorer using specific filter: `logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"`
*   **Interval:** Adjust `stack_trace_interval_seconds` to control trace collection frequency.

### Ahead of Time Compilation (AOT)

Pre-compile your training run for faster startup and OOM detection.

#### TPU Support

*   Compile using a CPU or a single VM.
*   Install `jax[tpu]` and other dependencies via `setup.sh`.

##### Example AOT:

```bash
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 global_parameter_scale=16 per_device_batch_size=4
```

##### Example AOT: Save compiled function, then load and run it
**Step 1: Run AOT and save compiled function**
```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```
**Step 2: Run train.py and load the compiled function**
```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```
#### GPU Support
##### Example
**Step 1: Run AOT and save compiled function**
```bash
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**
```bash
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```
### Automatically Upload Logs to Vertex Tensorboard
Follow [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload
Follow this [user guide](getting_started/Monitor_Goodput.md).

[**Visit the MaxText Repository on GitHub**](https://github.com/AI-Hypercomputer/maxtext)