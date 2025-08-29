# MaxText: High-Performance, Open-Source LLMs in Pure Python/JAX

**Unleash the power of large language models with MaxText, a high-performance and scalable open-source framework built for training and inference on TPUs and GPUs.** ([Original Repository](https://github.com/AI-Hypercomputer/maxtext))

## Key Features

*   **TPU & GPU Support:** Optimized for both Google Cloud TPUs and GPUs, enabling flexible hardware choices.
*   **Training & Inference:** Comprehensive support for both LLM training and inference workloads.
*   **Model Support:** Ready-to-use with a wide range of open models: Llama 2, Llama 3, Llama 4, Mistral, Mixtral, Gemma, DeepSeek, and Qwen3 families.
*   **High Performance:** Achieves impressive Model Flops Utilization (MFU) thanks to JAX and XLA compiler.
*   **Scalability:** Designed to scale from single-host setups to massive clusters.
*   **Modular Imports & Advanced Features:**  Supports modular imports, stack trace collection, Ahead-of-Time compilation, and Vertex AI Tensorboard integration.

## What's New
Stay updated with the latest developments:

*   **[August 13, 2025]**  Support for the Qwen3 MoE family of models.
*   **[July 27, 2025]**  Updated TFLOPS/s calculation to account for causal attention.
*   **[July 16, 2025]**  Repository restructuring for improved organization.
*   **[July 11, 2025]**  Multi-Token Prediction (MTP) training support.
*   **[June 25, 2025]** DeepSeek R1-0528 variant support.
*   **[April 24, 2025]** Llama 4 Maverick models supported.
*   **[April 14, 2025]** Llama 4 Scout models supported.
*   **[April 7, 2025]** Modular imports enabled with API change.
*   **[April 2, 2025]** DeepSeek v3-0324 variant support.
*   **[March 24, 2025]** DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]** Gemma 3 models (4B, 12B, 27B) support.
*   **[February, 2025]** Preview of building Maxtext Docker images using the JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Jumpstart your LLM projects with MaxText!  Follow the [First Run](getting_started/First_run.md) instructions to get up and running quickly.

Explore our comprehensive guides for training and inference with various open models:

*   [Gemma](https://ai.google.dev/gemma):  Fine-tune and decode with Google's open-weights LLM.
*   [Llama 2](https://llama.meta.com/llama2/): Train and run inference using Meta's popular Llama 2.
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Leverage Mistral AI's MoE models.
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Utilize DeepSeek AI's advanced MoE models, including support for pre-training, fine-tuning, and decoding.

For further details, explore the end-to-end tests and unit tests for more insights into MaxText's capabilities.

## Runtime Performance Results

Achieve cutting-edge performance with MaxText!  Reproduce the results from [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText offers a compelling alternative to existing LLM implementations.  It distinguishes itself through:

*   **High Performance & Scalability:**  Compared to MinGPT/NanoGPT, MaxText provides significantly higher MFU and robust scaling capabilities.
*   **Pure Python/JAX:**  Unlike Megatron-LM (which uses a mix of Python and CUDA), MaxText relies on pure Python and the XLA compiler.
*   **Emphasis on Simplicity:**  In contrast to Pax, MaxText favors direct code modification to encourage experimentation and customization.

## Features and Diagnostics

Optimize your MaxText experience with these helpful features:

### Collect Stack Traces

Capture stack traces to diagnose SPMD job issues:

1.  Set `collect_stack_trace: True` to enable stack trace collection.
2.  Configure stack trace output using `stack_trace_to_cloud: False` (console) or `stack_trace_to_cloud: True` (Cloud Logging).
3.  Adjust the collection frequency with `stack_trace_interval_seconds`.

See the related PyPI package: https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)

Improve startup times and identify potential OOM issues using AOT compilation with the `train_compile.py` tool.

#### TPU Support

Compile for TPUs using a CPU or a single VM from a different family. Install `jax[tpu]` and follow these examples:

##### Example AOT 1: Compile ahead of time basics

```bash
# Run on a single machine, e.g. a CPU
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

##### Example AOT 2: Save compiled function, then load and run it

**Step 1: Run AOT and save compiled function**

```bash
# Run on a single machine, e.g. a CPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
# Run on each host of the target hardware, e.g. each host on 2 slices of v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

#### GPU Support

GPU support for AOT compilation has some differences:

1.  A single GPU host is still required for compilation.
2.  For A3 Cloud GPUs, `compile_topology_num_slices` represents the number of machines to precompile for.

##### Example

**Step 1: Run AOT and save compiled function**

```bash
# Run on a single A3 machine
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
# Run on each of the 4 target A3 hosts.
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Integrate with Vertex AI Tensorboard for comprehensive monitoring; see the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for details.

### Monitor Goodput of Your Workload

Track the performance of your workload; refer to the [user guide](getting_started/Monitor_Goodput.md) for setup instructions.