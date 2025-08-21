# MaxText: High-Performance, Open-Source LLM Training and Inference

**Maximize your LLM potential with MaxText, a powerful, open-source framework for training and deploying large language models, built for speed and scalability.** ([Original Repository](https://github.com/AI-Hypercomputer/maxtext))

MaxText is a high-performance, highly scalable, open-source LLM written in pure Python/Jax, designed for both training and inference on Google Cloud TPUs and GPUs. It achieves impressive model flops utilization (MFU) and scales seamlessly from single hosts to massive clusters, all while maintaining simplicity and optimization-free performance thanks to Jax and the XLA compiler.

**Key Features:**

*   **TPU and GPU Support:** Train and run LLMs on both TPUs and GPUs for flexibility and cost-effectiveness.
*   **Training and Inference:**  Comprehensive support for both training and inference workloads.
*   **Model Compatibility:**  Supports a wide range of popular LLM families, including Llama 2, Llama 3, Mistral, Mixtral, Gemma, DeepSeek, and Qwen3 models.
*   **Modular Imports:** Utilize modular imports for improved organization and clarity.
*   **Ahead-of-Time Compilation:** Optimize training runs with Ahead-of-Time (AOT) compilation for fast startup and efficient resource utilization.
*   **Vertex AI TensorBoard Integration:** Automatically upload logs to Vertex AI TensorBoard for comprehensive monitoring and analysis.
*   **Goodput Monitoring:** Track and optimize the goodput of your workload.

**Announcements:**

*   [August 13, 2025] The Qwen3 MoE family of models is now supported.
*   [July 27, 2025] TFLOPS/s calculation has been updated to account for causal attention.
*   [July 16, 2025] Repository is being restructured for improved organization and clarity.
*   [July 11, 2025] Multi-Token Prediction (MTP) training is now supported.
*   [June 25, 2025] DeepSeek R1-0528 variant is now supported.
*   [April 24, 2025] Llama 4 Maverick models are now supported.
*   [April 14, 2025] Llama 4 Scout models are now supported.
*   **[April 7, 2025] 🚨🚨🚨 Support for modular imports added, introducing an API change for `train.py`.**
*   [April 2, 2025] DeepSeek v3-0324 variant is now supported.
*   [March 24, 2025] Support for DeepSeek v3 (671B) and v2-Lite (16B).
*   [March 12, 2025] Support for Gemma 3: 4B, 12B, and 27B.
*   [February, 2025] Preview of building Maxtext Docker images using JAX AI Training Images.

**Table of Contents:**

*   [Getting Started](getting_started/First_run.md)
*   [Runtime Performance Results](#runtime-performance-results)
*   [Comparison To Alternatives](#comparison-to-alternatives)
*   [Development](#development)
*   [Features and Diagnostics](#features-and-diagnostics)

**Getting Started**

Get up and running with MaxText quickly using our detailed [instructions](getting_started/First_run.md).  We provide comprehensive guides for various open models:

*   [Gemma](https://ai.google.dev/gemma): Training and fine-tuning instructions.
*   [Llama2](https://llama.meta.com/llama2/): Decode and fine-tuning instructions.
*   [Mixtral](https://mistral.ai/news/mixtral-of-experts/): Decode and fine-tuning instructions.
*   [DeepSeek](https://api-docs.deepseek.com/news/news1226): Pre-training, fine-tuning, and decoding instructions.

Explore the [end_to_end](end_to_end) directory for a full suite of tests. Continuous [unit tests](.github/workflows/RunTests.yml) are also available.

**Runtime Performance Results**

Find detailed performance results and reproduction instructions in [MaxText/configs/README.md](MaxText/configs/README.md).

**TPU v5p**

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

**TPU v5e**

See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

**Comparison to Alternatives**

MaxText draws inspiration from MinGPT/NanoGPT and is comparable to Nvidia's Megatron-LM and Google's Pax. It offers high performance, scalability, and a flexible development environment. MaxText's architecture differs by providing pure Python code optimized via XLA, which achieves comparable MFU to Megatron-LM.  MaxText distinguishes itself from Pax through its simpler, more direct approach to model implementation, making it an excellent choice for developers who prefer to modify the codebase directly.

**Features and Diagnostics**

*   **Collect Stack Traces:**  Enable stack trace collection to aid debugging by setting `collect_stack_trace: True`.
*   **Ahead of Time Compilation (AOT):**  Compile your training run ahead of time for improved performance and faster startup times with `train_compile.py`.  AOT compilation is supported for both TPUs and GPUs.
*   **Automatically Upload Logs to Vertex Tensorboard:** Integrate with Vertex AI TensorBoard for comprehensive monitoring.  See [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).
*   **Monitor Goodput of Your Workload:** Track Goodput metrics using the [user guide](getting_started/Monitor_Goodput.md).