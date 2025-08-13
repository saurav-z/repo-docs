# MaxText: High-Performance, Open-Source LLM Training and Inference

**Maximize your LLM performance with MaxText, a high-performance, scalable, open-source framework built in pure Python/Jax for training and inference on Google Cloud TPUs and GPUs.**  [Visit the original repo](https://github.com/AI-Hypercomputer/maxtext)

**Key Features:**

*   **High Performance:** Achieves industry-leading Model Flops Utilization (MFU) for fast training and inference.
*   **Scalable:** Designed to scale from single devices to massive clusters, enabling training of the largest language models.
*   **Open Source:**  Built on open-source principles, MaxText empowers users to experiment, customize, and contribute.
*   **TPU and GPU Support:**  Optimized for Google Cloud TPUs and GPUs, providing flexibility in hardware selection.
*   **Training and Inference:** Supports both training and inference workloads, providing a comprehensive solution.
*   **Model Support:**  Includes pre-configured support for a wide range of popular LLMs, including Llama 2, Llama 3, Mistral, Mixtral, Gemma, and DeepSeek.

**Latest Announcements:**

*   **July 27, 2025:** Attention flop calculations updated to account for causal, sliding window, and chunked attention. See [PR 1988](https://github.com/AI-Hypercomputer/maxtext/pull/1988), [PR 2009](https://github.com/AI-Hypercomputer/maxtext/pull/2009), and [PR 2030](https://github.com/AI-Hypercomputer/maxtext/pull/2030).
*   **July 16, 2025:** Repository restructuring for improved organization. See [RESTRUCTURE.md](RESTRUCTURE.md).
*   **July 11, 2025:** Multi-Token Prediction (MTP) training support.
*   **June 25, 2025:** DeepSeek R1-0528 variant support.
*   **April 24, 2025:** Llama 4 Maverick models support.
*   **April 14, 2025:** Llama 4 Scout models support.
*   **April 7, 2025:** Modular imports support and API changes (see details in the original README).
*   **April 2, 2025:** DeepSeek v3-0324 variant support.
*   **March 24, 2025:** DeepSeek v3 (671B) and v2-Lite (16B) support.
*   **March 12, 2025:** Gemma 3 (4B, 12B, and 27B) support.
*   **February, 2025:** (Preview) Building MaxText Docker images using JAX AI Training Images.

**Table of Contents:**

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

Get up and running quickly with MaxText using the [First Run instructions](getting_started/First_run.md). Explore the [getting_started](getting_started) folder for comprehensive guides.

**Model-Specific Guides:**

*   [Gemma (generations 1-3)](https://ai.google.dev/gemma)
*   [Llama2](https://llama.meta.com/llama2/)
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/)
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226)

## Runtime Performance Results

For detailed results, see [MaxText/configs/README.md](MaxText/configs/README.md).

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

See [MaxText/configs/v5e/](MaxText/configs/v5e/) for complete run configurations.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText offers a powerful alternative to other LLM implementations.  It draws inspiration from projects like MinGPT/NanoGPT, but supports industry-standard models. It also competes with Megatron-LM and PaxML.

*   **MinGPT/NanoGPT:** MaxText provides higher MFU and is massively scalable.
*   **Megatron-LM:** MaxText achieves comparable MFU.  MaxText is pure Python and relies heavily on XLA, while Megatron-LM uses CUDA kernels.
*   **PaxML:** MaxText encourages direct modification of the source code.

## Features and Diagnostics

### Collect Stack Traces

Enable stack trace collection for debugging by setting `collect_stack_trace: True` in `MaxText/configs/base.yml`. Set `stack_trace_to_cloud: True` to upload traces to cloud logging.

*   `stack_trace_interval_seconds`: Controls the interval for stack trace collection.

### Ahead of Time Compilation (AOT)

Use the `train_compile.py` tool to pre-compile `train_step` for faster startup and reduced OOM issues.

*   **TPU Support:** Use a CPU or single VM for compilation.
*   **GPU Support:** Supports single-host and multi-host compilation.

**AOT Example: Compile and save**

```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**AOT Example: Load and Run**

```bash
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

### Automatically Upload Logs to Vertex Tensorboard

Follow the [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to integrate Tensorboard.

### Monitor Goodput of Your Workload

Follow this [user guide](getting_started/Monitor_Goodput.md)