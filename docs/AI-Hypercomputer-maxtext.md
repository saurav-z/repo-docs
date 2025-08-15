# MaxText: High-Performance, Scalable LLM Training and Inference

**MaxText is a cutting-edge, open-source LLM solution built with pure Python/JAX, designed for training and inference on Google Cloud TPUs and GPUs.** Explore the [MaxText repository](https://github.com/AI-Hypercomputer/maxtext) for more details.

## Key Features

*   **High Performance:** Achieves impressive Model Flops Utilization (MFU) thanks to JAX and XLA.
*   **Scalability:** Designed to scale seamlessly from single hosts to massive clusters.
*   **Open Source:** Built with pure Python/Jax, providing flexibility and transparency.
*   **Training and Inference:** Supports both training and inference workflows.
*   **Model Support:** Compatible with a wide range of popular models including Llama 2/3/4, Mistral/Mixtral, Gemma, DeepSeek, and Qwen3 families.
*   **TPU and GPU Support:** Optimized for Google Cloud TPUs and GPUs.
*   **Modular Imports:** Includes an API change for `train.py`, now invoked via `python3 -m MaxText.train MaxText/configs/base.yml run_name=...`.

## Recent Updates

*   **[August 13, 2025]**: Added support for the Qwen3 MoE family of models (Qwen3-235B-A22B-Thinking-2507).
*   **[July 27, 2025]**: Updated TFLOPS/s calculations for causal, sliding window, and chunked attention.
*   **[July 16, 2025]**: Repository restructuring is planned.
*   **[July 11, 2025]**: Multi-Token Prediction (MTP) training is now supported.
*   **[June 25, 2025]**: Added support for DeepSeek R1-0528.
*   **[April 24, 2025]**: Llama 4 Maverick models are now supported.
*   **[April 14, 2025]**: Llama 4 Scout models are now supported.
*   **[April 7, 2025]**: Modular imports are now supported.
*   **[April 2, 2025]**: DeepSeek v3-0324 variant is now supported.
*   **[March 24, 2025]**: Support for DeepSeek v3 (671B) and v2-Lite (16B).
*   **[March 12, 2025]**: Added support for Gemma 3 (4B, 12B, and 27B).
*   **[February, 2025]**: (Preview): MaxText Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get up and running quickly with our [first run instructions](getting_started/First_run.md).

Explore our guides for specific model implementations:

*   [Gemma](https://ai.google.dev/gemma): Fine-tune and decode Gemma models.
*   [Llama2](https://llama.meta.com/llama2/): Instructions for decoding and fine-tuning.
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Fine-tuning and decoding guides.
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Pre-training, fine-tuning, and decoding.

Additional resources:

*   Full suite of end-to-end tests in [end_to_end](end_to_end).
*   Continuous [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

For detailed performance metrics, see [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText draws inspiration from projects like MinGPT/NanoGPT, while aiming for significantly higher MFU. It shares similarities with Nvidia's Megatron-LM, and Paxml. MaxText's design prioritizes pure Python/JAX for high performance, relying on the XLA compiler for optimization and providing a flexible framework for LLM development.

## Features and Diagnostics

### Collect Stack Traces

To enable debugging, configure the following in `MaxText/configs/base.yml`:

1.  `collect_stack_trace: True` to enable stack trace collection.
2.  `stack_trace_to_cloud: False` to display traces on the console or `stack_trace_to_cloud: True` to store in Cloud Logging via temporary files on TPUs.
3.  `stack_trace_interval_seconds` defines the interval between trace collections.

### Ahead of Time Compilation (AOT)

Use the `train_compile.py` tool to pre-compile your training run, for faster startup and restart times.
See examples below for TPUs and GPUs:

#### TPU Support

##### Example AOT 1: Compile ahead of time basics
```bash
# Run the below on a single machine, e.g. a CPU
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

##### Example AOT 2: Save compiled function, then load and run it
```bash
# Step 1: Run AOT and save compiled function
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3

# Step 2: Run train.py and load the compiled function
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

#### GPU Support

##### Example
```bash
# Step 1: Run AOT and save compiled function
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3

# Step 2: Run train.py and load the compiled function
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Configure automatic log uploads to Tensorboard in Vertex AI following the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

Follow this [user guide](getting_started/Monitor_Goodput.md) to monitor goodput.