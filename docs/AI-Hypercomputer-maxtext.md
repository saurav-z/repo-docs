# MaxText: High-Performance, Open-Source LLM Training and Inference

**Maximize your LLM performance with MaxText, a cutting-edge, open-source LLM framework optimized for speed and scalability on TPUs and GPUs.** ([Original Repository](https://github.com/AI-Hypercomputer/maxtext))

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/RunTests.yml)

## Key Features:

*   **High Performance:** Achieve exceptional Model Flops Utilization (MFU) thanks to pure Python/Jax and the XLA compiler. See impressive [Runtime Performance Results](#runtime-performance-results) below.
*   **Scalable:** Train and infer LLMs from single-host to large clusters of Google Cloud TPUs and GPUs.
*   **Open Source:**  Freely experiment, customize, and contribute to your LLM projects.
*   **TPU and GPU Support:** Runs efficiently on both hardware platforms.
*   **Training and Inference:** Supports both training and inference workflows.
*   **Model Support:** Pre-configured for popular LLMs, including:
    *   Llama 2, Llama 3, Llama 4 (Maverick and Scout models)
    *   Mistral and Mixtral family
    *   Gemma, Gemma 2, Gemma 3
    *   DeepSeek family
*   **Modular Imports:** Improved code organization with modular imports.

## Announcements

*   [July 27, 2025] Updated TFLOPS/s calculations to account for causal, sliding window, and chunked attention, improving accuracy.
*   [July 16, 2025] Repository restructuring for improved organization is underway.
*   [July 11, 2025] Multi-Token Prediction (MTP) training support for enhanced efficiency.
*   [June 25, 2025] DeepSeek R1-0528 variant is now supported!
*   [April 24, 2025] Llama 4 Maverick models are now supported!
*   [April 14, 2025] Llama 4 Scout models are now supported (8k context length).
*   **[April 7, 2025] API change for `train.py`: Use `python3 -m MaxText.train MaxText/configs/base.yml run_name=...` now.**
*   [April 2, 2025] DeepSeek v3-0324 variant is now supported!
*   [March 24, 2025] DeepSeek v3 (671B) and v2-Lite (16B) support for both TPUs and GPUs.
*   [March 12, 2025] Gemma 3 (4B, 12B, 27B) text-only support.
*   [February, 2025] (Preview): Build MaxText Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Begin your MaxText journey with detailed [instructions](getting_started/First_run.md) for your first run.

Explore guides for training and inference with various open models in the [getting started](getting_started) folder:

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

Also, find additional information in the [end_to_end](end_to_end) tests and [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

Reproduce these results with details in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText differentiates itself through its focus on performance, scalability, and code simplicity. MaxText achieves high MFU compared to similar implementations. In comparison to Megatron-LM, the codebases differ in programming strategies, MaxText is pure Python relying on the XLA compiler. Compared to Pax, MaxText is a simple, concrete implementation of various LLMs that encourages users to extend by forking and directly editing the source code.

## Development

[Original Repository](https://github.com/AI-Hypercomputer/maxtext)

## Features and Diagnostics

### Collect Stack Traces

Configure stack trace collection for debugging in `MaxText/configs/base.yml`:

1.  `collect_stack_trace: True` enables stack trace collection. Set to `False` to disable.
2.  `stack_trace_to_cloud: False` displays traces on the console. `True` stores traces in `/tmp/debugging` and uploads to Cloud Logging.
3.  `stack_trace_interval_seconds` sets the trace collection interval.

View traces in Cloud Logging using:

```
logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
jsonPayload.verb="stacktraceanalyzer"
```

For information on the PyPI package see: https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile for target hardware (TPUs and GPUs):

*   Provides OOM detection.
*   Enables fast startup and restart times.
*   Install `jax[tpu]` (and other dependencies) using `setup.sh`.

**AOT Example 1: Compile basics**

```bash
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

**AOT Example 2: Save/load compiled function**

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

**GPU Support**

*   Single GPU host required for compilation.
*   `compile_topology_num_slices` represents the number of A3 machines.

**GPU Example**

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

See [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

See [user guide](getting_started/Monitor_Goodput.md).