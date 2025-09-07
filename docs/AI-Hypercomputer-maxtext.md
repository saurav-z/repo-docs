# MaxText: Train and Inference Large Language Models with High Performance

**Maximize your LLM training and inference efficiency with MaxText, a high-performance, scalable, and open-source framework built in pure Python/Jax.**  [Explore the MaxText repository](https://github.com/AI-Hypercomputer/maxtext).

## Key Features

*   **High Performance & Scalability:** Achieve industry-leading Model Flops Utilization (MFU) and scale from single hosts to very large clusters, thanks to Jax and XLA.
*   **TPU & GPU Support:** Optimized for both Google Cloud TPUs and GPUs.
*   **Training & Inference:** Comprehensive support for both training and inference workflows.
*   **Model Compatibility:**  Supports a wide range of models including Llama 2, Llama 3, Mistral and Mixtral families, Gemma, Gemma 2 & 3, DeepSeek, Qwen3 (Dense and MoE), and more.
*   **Open Source & Flexible:**  Designed as a launching pad for LLM projects, encouraging modification and experimentation.

## Announcements

*   **[August 13, 2025]**:  Added support for the Qwen3 MoE family of models, starting with Qwen3-235B-A22B-Thinking-2507.
*   **[July 27, 2025]**: Updated TFLOPS/s calculations to account for causal, sliding window, and chunked attention.
*   **[July 16, 2025]**: Repository restructuring planned; review the proposed structure [here](RESTRUCTURE.md).
*   **[July 11, 2025]**: Introduced Multi-Token Prediction (MTP) training for enhanced efficiency.
*   **[June 25, 2025]**: Added support for the DeepSeek R1-0528 variant.
*   **[April 24, 2025]**:  Llama 4 Maverick models are now supported!
*   **[April 14, 2025]**: Llama 4 Scout models are now supported with context length up to 8k.
*   **[April 7, 2025]**:  **API change for `train.py`: Use `python3 -m MaxText.train ...` for modular imports.**  For the old behavior, use an older commit (`git checkout pre-module-v0.1.0`).
*   **[April 2, 2025]**: Added support for the DeepSeek v3-0324 variant.
*   **[March 24, 2025]**:  Support for DeepSeek v3 (671B) and v2-Lite (16B) models on TPUs and GPUs.
*   **[March 12, 2025]**:  Support for Gemma 3: 4B, 12B, and 27B models.
*   **[February, 2025] (Preview):** Preview of building MaxText Docker images using JAX AI Training Images. Learn more [Here](getting_started/Run_MaxText_via_xpk.md)

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get up and running quickly with MaxText using the [First Run instructions](getting_started/First_run.md). Explore the [getting started](getting_started) folder for detailed guides on training and inference.

**Key Model Guides:**

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma): Learn how to run decode and finetuning using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md), gemma2 and gemma3 scripts can be found [here](end_to_end/tpu/gemma2) and [here](end_to_end/tpu/gemma3).
*   [Llama2](https://llama.meta.com/llama2/): Instructions for decode and finetuning can be found [here](getting_started/Run_Llama2.md).
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Instructions for decode and finetuning can be found [here](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Explore pre-training, finetuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Find comprehensive end-to-end tests in the [end_to_end](end_to_end) directory. Unit tests are available [here](.github/workflows/RunTests.yml).

## Runtime Performance Results

Find performance metrics for both TPU v5p and v5e:

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

See full run configs in [src/MaxText/configs/v5e/](src/MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

More details on reproducing these results can be found in [src/MaxText/configs/README.md](src/MaxText/configs/README.md).

## Comparison to Alternatives

MaxText stands apart through its pure-Python/Jax implementation and focus on achieving high MFU, similar to Megatron-LM but with a different development strategy. It is also designed to be user-friendly, unlike Pax, and allows users to extend the model by forking the code and making changes.

## Features and Diagnostics

### Collect Stack Traces

Configure stack trace collection in `src/MaxText/configs/base.yml` to debug SPMD jobs:

1.  Set `collect_stack_trace: True` to enable stack trace collection.
2.  `stack_trace_to_cloud: True` uploads traces to Cloud Logging; `stack_trace_to_cloud: False` displays traces on the console.
3.  Adjust `stack_trace_interval_seconds` to control the frequency of trace collection.

The related PyPI package is located at: https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to pre-compile the `train_step` for faster startup and to identify OOM issues.

#### TPU Support

*   Install `jax[tpu]` and other dependencies using `setup.sh`.
*   Example compilation command: `python3 -m MaxText.train_compile ...`
*   Example of saving compiled function:  `python3 -m MaxText.train_compile ...`
*   Example of loading compiled function: `python3 -m MaxText.train ... compiled_trainstep_file=my_compiled_train.pickle ...`

#### GPU Support

*   GPU AOT compilation requires a single GPU host for compilation.
*   For A3 Cloud GPUs, `compile_topology_num_slices` represents the number of machines to precompile for.
*   Example: `python3 -m MaxText.train_compile ...`
*   Example: `python3 -m MaxText.train ... compiled_trainstep_file=my_compiled_train.pickle ...`

### Automatically Upload Logs to Vertex Tensorboard

Enable automatic upload of logs to Vertex AI Tensorboard.  See [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for more information.

### Monitor Goodput of Your Workload

Follow this [user guide](getting_started/Monitor_Goodput.md) to monitor goodput metrics.