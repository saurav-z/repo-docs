# MaxText: High-Performance, Scalable LLM Training & Inference in Pure Python/JAX

**Maximize your LLM training and inference with MaxText, an open-source, high-performance framework built in pure Python/JAX, optimized for Google Cloud TPUs and GPUs.**  [See the original repository](https://github.com/AI-Hypercomputer/maxtext)

## Key Features

*   **High Performance:** Achieve industry-leading Model Flops Utilization (MFU) for efficient training and inference.
*   **Scalability:** Train and run LLMs from a single host to massive clusters on TPUs and GPUs.
*   **Open Source:**  Leverage a flexible and customizable framework built in pure Python/JAX.
*   **TPU & GPU Support:** Optimized for Google Cloud TPUs (v5e, v5p) and GPUs (A100, H100).
*   **Training & Inference:** Support both training and inference workflows.
*   **Model Support:** Extensive support for popular LLMs including Llama 2, Llama 3, Gemma, DeepSeek, Qwen3, and Mixtral families.

## Announcements

*   **[September 5, 2025]**  MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md). Ensure you run `pip install -e .` from the MaxText root.
*   **[August 13, 2025]** Added support for the Qwen3 MoE family of models, starting with Qwen3-235B-A22B-Thinking-2507.
*   **[July 27, 2025]** Updated TFLOPS/s calculation to account for causal, sliding window, and chunked attention (see [PRs](https://github.com/AI-Hypercomputer/maxtext/pull/1988), [2009](https://github.com/AI-Hypercomputer/maxtext/pull/2009), [2030](https://github.com/AI-Hypercomputer/maxtext/pull/2030)).
*   **[July 16, 2025]** Restructuring for improved organization.  [RESTRUCTURE.md](RESTRUCTURE.md).
*   **[July 11, 2025]** Multi-Token Prediction (MTP) training is now supported!
*   **[June 25, 2025]** DeepSeek R1-0528 variant is now supported!
*   **[April 24, 2025]** Llama 4 Maverick models are now supported!
*   **[April 14, 2025]** Llama 4 Scout models are now supported.
*   **[April 7, 2025]**  🚨🚨🚨  Modular imports are supported. Use `python3 -m MaxText.train MaxText/configs/base.yml run_name=...`. For older API, use `git checkout pre-module-v0.1.0`.
*   **[April 2, 2025]** DeepSeek v3-0324 variant is now supported!
*   **[March 24, 2025]** Support for DeepSeek v3 (671B) and v2-Lite (16B).
*   **[March 12, 2025]** Support for Gemma 3: 4B, 12B, and 27B.
*   **[February, 2025]** (Preview): Building Maxtext Docker images using JAX AI Training Images. Learn more [Here](getting_started/Run_MaxText_via_xpk.md)

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get up and running quickly with our [First Run instructions](getting_started/First_run.md).

### Model Examples
Follow user guides in the [getting started](getting_started) folder to know more.

*   **Gemma:** Learn to run decode and finetuning using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md). For Gemma 2 and 3, use the corresponding [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts.
*   **Llama2:** Use [these instructions](getting_started/Run_Llama2.md) for decoding and finetuning.
*   **Mixtral:** Instructions for decoding and finetuning are available [here](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:** Run pre-training, finetuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

For further information, check out the [end_to_end](end_to_end) directory for examples and the [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

Refer to [src/MaxText/configs/README.md](src/MaxText/configs/README.md) for reproduction details.

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

MaxText distinguishes itself with its pure Python/JAX implementation, offering a balance of performance and flexibility compared to alternatives like MinGPT/NanoGPT, Megatron-LM, and Pax. It focuses on scaling to very large clusters, and MFU, while remaining a simple implementation that encourages users to experiment and extend.

## Features and Diagnostics

### Collect Stack Traces

Configure `collect_stack_trace`, `stack_trace_to_cloud`, and `stack_trace_interval_seconds` in `src/MaxText/configs/base.yml` to enable stack trace collection for debugging SPMD jobs. View traces in Cloud Logging using the query provided.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` for ahead-of-time compilation for faster startup times and to catch OOM errors early.

#### TPU Support

*   Use a CPU or single VM to pre-compile for a TPU cluster.
*   Install `jax[tpu]` and run `setup.sh`.

##### Example AOT 1: Compile ahead of time basics

```bash
python3 -m MaxText.train_compile src/MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

##### Example AOT 2: Save compiled function, then load and run it

**Step 1: Run AOT and save compiled function**

```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile src/MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train src/MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

#### GPU Support

*   GPU AoT compilation requires a GPU host, but a single host can compile for a larger cluster.
*   For A3 Cloud GPUs, `compile_topology_num_slices` represents the number of machines.

##### Example

**Step 1: Run AOT and save compiled function**

```bash
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile src/MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train src/MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for instructions on automatic log uploads.

### Monitor Goodput of Your Workload

Consult the [user guide](getting_started/Monitor_Goodput.md) to monitor goodput metrics.