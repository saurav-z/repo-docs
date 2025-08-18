# MaxText: High-Performance, Scalable LLM Training and Inference

**MaxText is a cutting-edge, open-source Large Language Model (LLM) built in pure Python/Jax, optimized for training and inference on Google Cloud TPUs and GPUs, enabling state-of-the-art performance and scalability.**  Check out the original repository [here](https://github.com/AI-Hypercomputer/maxtext).

**Key Features:**

*   **TPU and GPU Support:** Optimized for Google Cloud TPUs (v5p, v5e) and GPUs (A3).
*   **Training and Inference:** Supports both model training and inference workflows.
*   **Model Compatibility:**  Supports a wide variety of models, including Llama 2, Llama 3, Mistral, Mixtral, Gemma, DeepSeek, and Qwen3 families.
*   **High Performance:** Achieves excellent Model Flops Utilization (MFU) for efficient training and inference.
*   **Scalability:** Designed to scale from single hosts to large clusters with minimal optimization effort, leveraging the power of JAX and XLA.
*   **Modular Imports & API Change:** Updated for modular imports, with a corresponding API change for `train.py`.

## Announcements

*   [August 13, 2025]  Qwen3 MoE family of models (Qwen3-235B-A22B-Thinking-2507) is now supported.
*   [July 27, 2025]  Updated TFLOPS/s calculation to account for causal, sliding window, and chunked attention.
*   [July 16, 2025]  Repository restructuring for improved organization and clarity.
*   [July 11, 2025]  Support for Multi-Token Prediction (MTP) training.
*   [June 25, 2025]  DeepSeek R1-0528 variant is now supported!
*   [April 24, 2025]  Llama 4 Maverick models are now supported!
*   [April 14, 2025]  Llama 4 Scout models are now supported.
*   **[April 7, 2025] 🚨🚨🚨 Modular imports API change: Invoke the script via `python3 -m MaxText.train MaxText/configs/base.yml run_name=...`.**
*   [April 2, 2025] DeepSeek v3-0324 variant is now supported!
*   [March 24, 2025] Support for DeepSeek v3 (671B) and v2-Lite (16B).
*   [March 12, 2025] Support for Gemma 3: 4B, 12B, and 27B in text-only formats.
*   [February, 2025] (Preview): Preview of building Maxtext Docker images using the JAX AI Training Images.

# Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

# Getting Started

Get started with MaxText by following the [First Run instructions](getting_started/First_run.md).  Explore detailed guides for training and inference of various open models within the [getting started](getting_started) folder.

## Model-Specific Guides:

*   [Gemma](https://ai.google.dev/gemma): Comprehensive guide for Gemma (Generations 1-3) training and decoding.
*   [Llama2](https://llama.meta.com/llama2/): Instructions for Llama 2 training and decoding.
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/):  Guide to running decode and finetuning Mixtral models.
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Instructions for pre-training, fine-tuning and decoding DeepSeek models.

Access comprehensive end-to-end tests in the [end_to_end](end_to_end) directory and review the continuous [unit tests](.github/workflows/RunTests.yml) to explore MaxText functionalities.

# Runtime Performance Results

Achieve state-of-the-art performance. Detailed performance results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

## TPU v5p

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

## TPU v5e

Performance results for various model sizes and hardware configurations. See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

# Comparison to Alternatives

MaxText builds upon the simplicity of projects like [MinGPT](https://github.com/karpathy/minGPT) and [NanoGPT](https://github.com/karpathy/nanoGPT), offering a more sophisticated solution with support for industry-standard models and superior scalability.  It achieves a significantly higher MFU compared to existing codebases and implements key-value caching for efficient autoregressive decoding.

Compared to projects such as [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Pax](https://github.com/google/paxml), MaxText's Python-first approach, powered by JAX and XLA, enables a balance of performance, scalability, and flexibility.  MaxText prioritizes direct code modification to empower users in adapting models for their specific needs.

# Features and Diagnostics

## Collect Stack Traces

Enable stack trace collection to debug SPMD jobs, identify program hang issues and analyze VM crashes.  Configure the following parameters in `MaxText/configs/base.yml`:
1.  Set `collect_stack_trace: True` to enable the stack trace collection.
2.  `stack_trace_to_cloud: True` uploads traces to Cloud Logging, which can be viewed in the Logs Explorer. Use the following query in the Logs Explorer on Cloud Logging:
    ```
    logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
    jsonPayload.verb="stacktraceanalyzer"
    ```
3.  Set `stack_trace_interval_seconds` (e.g., `600` seconds) to control the frequency of stack trace collection.

## Ahead of Time Compilation (AOT)

Optimize training with ahead-of-time compilation using the `train_compile.py` tool. Compile for TPUs and GPUs with detailed instructions and examples.

### TPU Support

Utilize a CPU or a single VM to pre-compile for TPU clusters. This enables two key benefits:

*   Identify out-of-memory (OOM) conditions early.
*   Save and load the compiled function for faster startup and restart times.

#### Example AOT 1: Compile ahead of time basics
```bash
# Run the below on a single machine, e.g. a CPU
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

#### Example AOT 2: Save compiled function, then load and run it
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

### GPU Support

AOT compilation is also supported for GPUs, with the following considerations:

1.  GPU does not support compilation across hardware. A single GPU host can compile for larger clusters of the same hardware.
2.  For [A3 Cloud GPUs](https://cloud.google.com/compute/docs/gpus#h100-gpus), the maximum "slice" size is a single host, and the `compile_topology_num_slices` parameter represents the number of A3 machines to precompile for.

#### Example
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

## Automatically Upload Logs to Vertex Tensorboard

Enable automatic upload of logs collected in a directory to a Tensorboard instance in Vertex AI. Follow [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

## Monitor Goodput of Your Workload

Monitor goodput metrics using the [user guide](getting_started/Monitor_Goodput.md).