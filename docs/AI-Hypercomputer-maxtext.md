# MaxText: High-Performance, Open-Source LLM Training and Inference

**Maximize your LLM performance with MaxText, a cutting-edge, open-source framework built in pure Python/Jax, optimized for Google Cloud TPUs and GPUs.** ([Original Repo](https://github.com/AI-Hypercomputer/maxtext))

## Key Features

*   **High Performance:** Achieves industry-leading Model Flops Utilization (MFU) for faster training and inference.
*   **Scalability:** Designed to scale from single host to very large clusters, leveraging the power of Jax and XLA.
*   **Open Source:** Fully open-source, allowing for customization and community contributions.
*   **TPU & GPU Support:** Compatible with both Google Cloud TPUs and GPUs.
*   **Training and Inference:** Supports both training and inference workloads.
*   **Model Compatibility:**  Supports a wide range of models, including Llama 2, Llama 3, Mistral and Mixtral family, Gemma, Gemma 2, Gemma 3, DeepSeek, Qwen3 Dense and MoE family.

## What's New

*   **[August 13, 2025]** Added support for the Qwen3 MoE family of models.
*   **[July 27, 2025]** Updated TFLOPS/s calculation for causal attention.
*   **[July 16, 2025]** Repository restructuring for improved organization.
*   **[July 11, 2025]** Added Multi-Token Prediction (MTP) training support.
*   **[June 25, 2025]** Added support for DeepSeek R1-0528.
*   **[April 24, 2025]** Llama 4 Maverick models are now supported!
*   **[April 14, 2025]** Llama 4 Scout models are now supported.
*   **[April 7, 2025]** Modular imports supported!
*   **[April 2, 2025]** DeepSeek v3-0324 variant is now supported!
*   **[March 24, 2025]** DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]** Gemma 3 support (4B, 12B, and 27B).
*   **[February, 2025]** Preview: Building MaxText Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get started quickly with our comprehensive [getting started](getting_started) guides, including:

*   **Gemma:** Decode and finetune using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md).
*   **Llama2:** Decode and finetune using [these instructions](getting_started/Run_Llama2.md).
*   **Mixtral:** Decode and finetune using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:** Pre-training, finetuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Explore the end-to-end tests in the [end_to_end](end_to_end) directory.

## Runtime Performance Results

Achieve superior performance with MaxText.  See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

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

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText offers significant advantages:

*   **MinGPT/NanoGPT:** MaxText is more complex and scalable, supporting industry-standard models and scaling to tens of thousands of chips.
*   **Nvidia/Megatron-LM:** MaxText achieves comparable MFUs with a different programming strategy (pure Python with XLA).
*   **Pax:** MaxText provides a simple, concrete implementation, encouraging users to extend by forking and directly editing the source code.

## Features and Diagnostics

### Collect Stack Traces

*   Set `collect_stack_trace: True` in `MaxText/configs/base.yml` to enable.
*   Stack traces can be displayed on console or stored in Cloud Logging with `stack_trace_to_cloud: True`.
*   Configure the collection interval with `stack_trace_interval_seconds`.
*   See the related PyPI package for more info: [cloud-tpu-diagnostics](https://pypi.org/project/cloud-tpu-diagnostics).

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile for your target hardware (TPUs or GPUs) for faster startup and restart times.

#### TPU Support

*   Compile using a CPU or a single VM.
*   OOM errors are flagged during compilation.

```bash
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

Save and load compiled functions:

```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3

# Load the compiled function during training:
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

#### GPU Support

*   Requires a GPU host for compilation, even for larger clusters.
*   Use `compile_topology_num_slices` to represent the number of machines for precompilation.

```bash
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3

# Run training using compiled function
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to upload logs.

### Monitor Goodput of Your Workload

Follow the [user guide](getting_started/Monitor_Goodput.md) for information on monitoring Goodput metrics.