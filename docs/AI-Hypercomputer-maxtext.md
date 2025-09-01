# MaxText: High-Performance, Open-Source LLMs with Jax

**Unleash the power of large language models with MaxText, a cutting-edge, open-source framework built for speed and scalability using pure Python/Jax.** ([Original Repo](https://github.com/AI-Hypercomputer/maxtext))

MaxText is designed for both research and production, enabling you to quickly experiment and build upon a robust foundation for your LLM projects.

## Key Features

*   **High Performance:** Achieves impressive Model Flops Utilization (MFU) through the power of Jax and the XLA compiler.
*   **Scalability:** Designed to scale from single host to massive clusters of TPUs and GPUs.
*   **Open-Source:** Fully open-source and written in pure Python/Jax, promoting transparency and community collaboration.
*   **TPU and GPU Support:** Optimized for both Google Cloud TPUs and GPUs, providing flexibility in hardware choice.
*   **Training and Inference:** Supports both training and inference workloads.
*   **Model Support:** Extensive support for popular open-source models, including Llama 2, Llama 3, Mistral, Mixtral, Gemma, Gemma 2, Gemma 3, DeepSeek, and Qwen3.
*   **Modular Imports:** Support for modular imports, with API changes.
*   **Multi-Token Prediction (MTP) Training:** Adds an auxiliary loss based on predicting multiple future tokens, inspired by the [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
*   **Ahead-of-Time Compilation (AOT):** Compile your training runs for fast startup and restart times.
*   **Diagnostics:** Collect stack traces and monitor the goodput metrics of your workload.
*   **Automated Vertex AI TensorBoard integration:** Log and visualize your results effectively

## Announcements

*   [August 13, 2025] - Qwen3 MoE family of models support
*   [July 27, 2025] - Updated TFLOPS/s calculation for causal and sliding window attention.
*   [July 16, 2025] - Restructuring of the MaxText repository for improved organization and clarity.
*   [July 11, 2025] - Multi-Token Prediction (MTP) training support.
*   [June 25, 2025] - DeepSeek R1-0528 variant support!
*   [April 24, 2025] - Llama 4 Maverick models support.
*   [April 14, 2025] - Llama 4 Scout models support.
*   [April 7, 2025] - Modular imports support: API change for `train.py`.
*   [April 2, 2025] - DeepSeek v3-0324 variant support.
*   [March 24, 2025] - DeepSeek v3 (671B) and v2-Lite (16B) support.
*   [March 12, 2025] - Gemma 3 support (4B, 12B, 27B).
*   [February, 2025] (Preview): Maxtext Docker images preview

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get up and running with MaxText quickly by following the [First Run Instructions](getting_started/First_run.md).

Explore the [getting started](getting_started) folder for model-specific guides on training and inference.

Helpful resources:

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma): Instructions for decoding and finetuning models.
*   [Llama2](https://llama.meta.com/llama2/): Instructions for decoding and finetuning models.
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Instructions for decoding and finetuning models.
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Instructions for pre-training, finetuning, and decoding models.

For end-to-end tests, see the [end_to_end](end_to_end) folder. The [unit tests](.github/workflows/RunTests.yml) are also a good source for understanding MaxText.

## Runtime Performance Results

Details on reproducing these results are in [MaxText/configs/README.md](MaxText/configs/README.md).

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

See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText draws inspiration from projects like MinGPT and NanoGPT, but it is a more sophisticated LLM implementation, supporting industry-standard models and scaling to thousands of chips. MaxText has an MFU more than three times the reported with that codebase, is massively scalable and implements a key-value cache for efficient auto-regressive decoding.

MaxText is comparable to Nvidia's Megatron-LM, achieving similar MFUs. The two implementations highlight different programming strategies, with MaxText using pure Python/Jax and relying on the XLA compiler. Megatron-LM uses a mix of Python and CUDA, relying on optimized CUDA kernels.

MaxText is also similar to Pax, providing high-performance and scalable LLM implementations in Jax.  Pax focuses on enabling powerful configuration parameters, enabling developers to change the model by editing config parameters. By contrast, MaxText is a simple, concrete implementation of various LLMs that encourages users to extend by forking and directly editing the source code.

## Features and Diagnostics

### Collect Stack Traces

Enable stack trace collection to debug SPMD jobs running on accelerators, especially when errors or VM issues occur.  Adjust parameters in `MaxText/configs/base.yml`:

1.  Set `collect_stack_trace: True` to enable stack trace collection. Disable with `collect_stack_trace: False`.
2.  `stack_trace_to_cloud: False` displays traces on the console.  `stack_trace_to_cloud: True` stores traces in `/tmp/debugging` and uploads them to Cloud Logging. View in Logs Explorer using:
    ```
    logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
    jsonPayload.verb="stacktraceanalyzer"
    ```
3.  `stack_trace_interval_seconds` sets the interval for stack trace collection (e.g., `600` for every 10 minutes).

Related PyPI package: [cloud-tpu-diagnostics](https://pypi.org/project/cloud-tpu-diagnostics).

### Ahead of Time Compilation (AOT)

Compile your training run ahead of time with `train_compile.py` to identify OOM errors and save compiled functions for faster startup.

*   Install `jax[tpu]` and other dependencies by running `setup.sh`.

#### TPU Support

Use a CPU or single VM for pre-compilation for TPUs.

##### Example AOT 1: Compile ahead of time basics

```bash
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

##### Example AOT 2: Save compiled function, then load and run it

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

AOT compilation is also supported for GPUs, with the following differences:

1.  A single GPU host can compile for a cluster of the same hardware.
2.  For A3 Cloud GPUs, `compile_topology_num_slices` represents the number of A3 machines to precompile for (single host maximum slice size).

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

MaxText supports automatic upload of logs to a Vertex AI Tensorboard instance. See [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

Follow this [user guide](getting_started/Monitor_Goodput.md) to monitor goodput metrics.