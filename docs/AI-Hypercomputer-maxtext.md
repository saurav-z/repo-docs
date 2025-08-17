# MaxText: High-Performance, Scalable LLM Training and Inference with JAX

**Maximize your LLM potential with MaxText, a cutting-edge, open-source framework built for high-performance training and inference on TPUs and GPUs.**  ([Original Repository](https://github.com/AI-Hypercomputer/maxtext))

MaxText is a powerful and flexible framework for training and deploying large language models, written in pure Python/Jax, designed to achieve high performance and scalability. Whether you're in research or production, MaxText offers a strong foundation for your LLM projects.

**Key Features:**

*   **TPU and GPU Support:** Optimized for both Google Cloud TPUs and GPUs, ensuring flexibility and performance.
*   **Training and Inference:** Comprehensive support for both model training and inference tasks.
*   **Open-Source & Customizable:** Built in Python/Jax, providing a simple and "optimization-free" approach that is easy to modify and extend.
*   **Model Compatibility:** Supports a wide range of popular models, including Llama 2, Llama 3, Mistral, Mixtral, Gemma, DeepSeek, and Qwen3.
*   **High Performance:** Achieves impressive Model Flops Utilization (MFU) and scales seamlessly from single host to very large clusters.

**What's New:**

*   [August 13, 2025] Support for the Qwen3 MoE family of models.
*   [July 27, 2025] Updated TFLOPS/s calculation.
*   [July 16, 2025] Repository restructuring for improved clarity.
*   [July 11, 2025] Multi-Token Prediction (MTP) training support.
*   [June 25, 2025] DeepSeek R1-0528 variant support.
*   [April 24, 2025] Llama 4 Maverick models are now supported!
*   [April 14, 2025] Llama 4 Scout models are now supported.
*   **[April 7, 2025] API Change: Modular imports are now supported.**
*   [April 2, 2025] DeepSeek v3-0324 variant support.
*   [March 24, 2025] DeepSeek v3 and v2-Lite support.
*   [March 12, 2025] Gemma 3 support.
*   [February, 2025] (Preview): Maxtext Docker images using JAX AI Training Images.

## Table of Contents

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get started quickly with our comprehensive [instructions](getting_started/First_run.md).

MaxText facilitates both training and inference of various open models. Explore the user guides in the [getting started](getting_started) directory to learn more.

Helpful guides:

*   **Gemma:** Instructions for running decode and finetuning ([gemma](end_to_end/tpu/gemma/Run_Gemma.md)). Also includes scripts for Gemma 2 ([gemma2](end_to_end/tpu/gemma2)) and Gemma 3 ([gemma3](end_to_end/tpu/gemma3)).
*   **Llama 2:** Instructions for running decode and finetuning ([Run_Llama2](getting_started/Run_Llama2.md)).
*   **Mixtral:** Instructions for running decode and finetuning ([Run_Mixtral](end_to_end/tpu/mixtral/Run_Mixtral.md)).
*   **DeepSeek:** Instructions for running pre-training, finetuning, and decoding ([Run_DeepSeek](end_to_end/tpu/deepseek/Run_DeepSeek.md)).

The [end_to_end](end_to_end) tests, which run nightly, and the continuous [unit tests](.github/workflows/RunTests.yml) are great resources for understanding and exploring MaxText capabilities.

## Runtime Performance Results

More detailed performance data can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

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

MaxText, inspired by [MinGPT](https://github.com/karpathy/minGPT) and [NanoGPT](https://github.com/karpathy/nanoGPT), stands out by supporting industry-standard models and scaling to tens of thousands of chips. Its MFU exceeds that of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Pax](https://github.com/google/paxml), and is over three times the reported performance of similar codebases. MaxText differs from these by focusing on simplicity via Jax, leveraging the XLA compiler.

## Features and Diagnostics

### Collect Stack Traces

Configure MaxText to collect stack traces for debugging, including on faults or hangs.

1.  Set `collect_stack_trace: True` to enable.
2.  Set `stack_trace_to_cloud: False` to view traces on the console or `True` to save traces to Cloud Logging via a temporary file.
3.  Set `stack_trace_interval_seconds` for the interval between trace collections.

### Ahead of Time Compilation (AOT)

Use `train_compile.py` to compile training runs for faster startup and to catch OOM issues early.

**TPU Support:**

*   Compile for TPU clusters using a single CPU or a single VM.
*   Install `jax[tpu]` and run `setup.sh` if necessary.

**Example AOT 1: Compile ahead of time basics**

```bash
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

**Example AOT 2: Save compiled function, then load and run it**

1.  **Run AOT and save compiled function**

```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

2.  **Run train.py and load the compiled function**

```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

**GPU Support:**

*   GPU compilation requires a GPU host.
*   `compile_topology_num_slices` represents the number of A3 machines.

**Example**

1.  **Run AOT and save compiled function**

```bash
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

2.  **Run train.py and load the compiled function**

```bash
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Automatically upload logs to a Tensorboard instance in Vertex AI. See the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).

### Monitor Goodput of Your Workload

Follow this [user guide](getting_started/Monitor_Goodput.md) for monitoring.