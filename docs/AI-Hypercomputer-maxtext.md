# MaxText: High-Performance, Open-Source LLM Training & Inference

**MaxText is a powerful, open-source framework enabling the efficient training and inference of large language models on Google Cloud TPUs and GPUs.** ([Original Repo](https://github.com/AI-Hypercomputer/maxtext))

## Key Features

*   **High Performance:** Achieves industry-leading Model Flops Utilization (MFU) for optimal resource usage. See performance results below.
*   **Scalability:** Designed to scale from single devices to very large clusters of TPUs and GPUs.
*   **Open-Source & Flexible:** Built with pure Python/Jax, allowing for easy customization and experimentation.
*   **TPU & GPU Support:** Supports training and inference on Google Cloud TPUs (v5p & v5e) and GPUs.
*   **Model Support:** Pre-configured to run Llama 2, Llama 3, Mistral, Mixtral, Gemma, DeepSeek, and Qwen3 models.

## Announcements

*   **[August 13, 2025]** Support for the Qwen3 MoE family of models has been added.
*   **[July 27, 2025]** TFLOPS/s calculations updated to account for causal, sliding window and chunked attention.
*   **[July 16, 2025]** Repository restructuring proposed for improved organization.
*   **[July 11, 2025]** Multi-Token Prediction (MTP) training support added for enhanced efficiency, inspired by the [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1).
*   **[June 25, 2025]** DeepSeek R1-0528 variant support added.
*   **[April 24, 2025]** Llama 4 Maverick models are now supported!
*   **[April 14, 2025]** Llama 4 Scout models are now supported.
*   **[April 7, 2025]** Modular imports supported with API change for `train.py`.
*   **[April 2, 2025]** DeepSeek v3-0324 variant support added.
*   **[March 24, 2025]** DeepSeek v3 (671B) and v2-Lite (16B) support added.
*   **[March 12, 2025]** Gemma 3 support (4B, 12B, 27B).
*   **[February, 2025] (Preview):** MaxText Docker image building is now available using the JAX AI Training Images, for both TPUs and GPUs.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Begin your MaxText journey with the [First Run](getting_started/First_run.md) guide.  Explore the guides for running, decoding, and fine-tuning various LLMs.

Explore these popular models:

*   **Gemma:** Family of open-weights LLMs by Google DeepMind. Decode and fine-tune using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md) and [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts for checkpoint convertion and decoding.
*   **Llama2:** Family of open-weights LLMs by Meta. Run decode and fine-tuning using [these instructions](getting_started/Run_Llama2.md).
*   **Mixtral:**  A family of open-weights sparse mixture-of-experts (MoE) models by Mistral AI. Run decode and fine-tuning using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
*   **DeepSeek:**  Family of open-weights MoE models by DeepSeek AI. Run pre-training, fine-tuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

Find additional guides in the [end_to_end](end_to_end) folder for end-to-end tests and [unit tests](.github/workflows/RunTests.yml).

## Runtime Performance Results

Detailed performance data is available in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText, inspired by [MinGPT](https://github.com/karpathy/minGPT)/[NanoGPT](https://github.com/karpathy/nanoGPT), offers superior performance and scalability, with an MFU exceeding the results reported by the latter by three times.

MaxText is comparable to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM),  achieving comparable MFU, yet differs in the reliance on Python and JAX, compared to a mix of Python and CUDA.

Similar to [Pax](https://github.com/google/paxml), MaxText provides high-performance and scalable LLM implementations in Jax, differing in its more direct approach encouraging users to extend it by forking and editing the source code.

## Features and Diagnostics

### Collect Stack Traces

Collect stack traces on faults by setting these configs in `MaxText/configs/base.yml`:

1.  `collect_stack_trace: True` to enable.
2.  `stack_trace_to_cloud: False` to display traces on console, `stack_trace_to_cloud: True` to store to cloud.
3.  `stack_trace_interval_seconds` for trace collection frequency.

### Ahead of Time Compilation (AOT)

Use the `train_compile.py` tool to pre-compile for faster startup and efficient resource management. Install the dependencies with the `setup.sh` script.

#### Example AOT 1: Compile ahead of time basics

```bash
# Run the below on a single machine, e.g. a CPU
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

#### Example AOT 2: Save compiled function, then load and run it

**Step 1: Run AOT and save compiled function**

```bash
# Run the below on a single machine, e.g. a CPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
# Run the below on each host of the target hardware, e.g. each host on 2 slices of v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### GPU Support

1.  Single host is required for AoT compilation.
2.  For [A3 Cloud GPUs](https://cloud.google.com/compute/docs/gpus#h100-gpus), the `compile_topology_num_slices` parameter is the number of machines to precompile for.

#### Example

**Step 1: Run AOT and save compiled function**

```bash
# Run the below on a single A3 machine
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

```bash
# Run the below on each of the 4 target A3 hosts.
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) for automatic log uploads.

### Monitor Goodput of Your Workload

Follow the [user guide](getting_started/Monitor_Goodput.md) to monitor workload goodput metrics.