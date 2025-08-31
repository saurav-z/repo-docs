# MaxText: High-Performance, Scalable LLM Training and Inference

**Unlock the power of large language models with MaxText, a high-performance, open-source framework for training and inferencing LLMs on TPUs and GPUs.**

[Link to Original Repo](https://github.com/AI-Hypercomputer/maxtext)

## Key Features

*   **Optimized Performance:** Achieve exceptional Model Flops Utilization (MFU) for efficient training and inference.
*   **Scalability:** Scale from single-host to massive clusters of TPUs and GPUs.
*   **Open Source & Flexible:** Built with pure Python/Jax, enabling easy customization and experimentation.
*   **Wide Model Support:** Supports training and inference for popular models: Llama 2, Llama 3, Mistral and Mixtral family, Gemma, Gemma 2, Gemma 3, DeepSeek, Qwen3 Dense and MoE family.
*   **TPU and GPU Compatibility:** Runs seamlessly on both Google Cloud TPUs and GPUs.
*   **Training and Inference:** Supports both training and inference pipelines.
*   **Modular Imports & Ahead-of-Time Compilation (AOT):** Faster and more efficient training runs.

## Announcements

*   **[August 13, 2025]** Added support for the Qwen3 MoE family of models.
*   **[July 27, 2025]** Updated TFLOPS/s calculation to account for causal, sliding window and chunked attentions.
*   **[July 16, 2025]** Restructuring MaxText repository.
*   **[July 11, 2025]** Support for Multi-Token Prediction (MTP) training.
*   **[June 25, 2025]** DeepSeek R1-0528 variant support.
*   **[April 24, 2025]** Llama 4 Maverick models support.
*   **[April 14, 2025]** Llama 4 Scout models support.
*   **[April 7, 2025]** Modular imports with API change.
*   **[April 2, 2025]** DeepSeek v3-0324 variant support.
*   **[March 24, 2025]** DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **[March 12, 2025]** Gemma 3: 4B, 12B, and 27B support.
*   **[February, 2025]** (Preview): Maxtext Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Dive into MaxText with our comprehensive [Getting Started](getting_started/First_run.md) guide.

Explore our user guides and leverage MaxText to train and fine-tune a variety of open models:

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

## Runtime Performance Results

See the impressive performance of MaxText on TPUs and GPUs.

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

MaxText offers a compelling alternative to existing LLM implementations:

*   **Compared to MinGPT/NanoGPT:** MaxText is more complex, supporting more industry standard models and scaling to tens of thousands of chips, offering more than three times the MFU.
*   **Compared to Megatron-LM:** MaxText achieves comparable MFUs, with the distinction of being pure Python and relying on XLA compiler for high performance.
*   **Compared to Pax:**  MaxText is a simpler, concrete implementation of various LLMs that encourages users to extend by forking and directly editing the source code, differing from Pax's config-driven approach.

## Features and Diagnostics

### Collect Stack Traces

Debug SPMD jobs with these configurations.

1.  Set `collect_stack_trace: True` to enable. Set `collect_stack_trace: False` to disable.
2.  Set `stack_trace_to_cloud: False` to display stack traces on console. `stack_trace_to_cloud: True` will create a temporary file in `/tmp/debugging`.
3.  `stack_trace_interval_seconds` to control interval (default 600 seconds).

### Ahead of Time Compilation (AOT)

Pre-compile for faster TPU and GPU training:

#### TPU Support

Use a CPU or single VM to pre-compile for a TPU cluster.
Install `jax[tpu]` and run `setup.sh`.

##### Example AOT 1: Compile ahead of time basics
```bash
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
GPU AOT compilation instructions.

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

See [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to know more.

### Monitor Goodput of Your Workload

See [user guide](getting_started/Monitor_Goodput.md) to know more.