# Checkpoint Engine: Accelerating LLM Weight Updates for Efficient Inference

**Checkpoint Engine** is a lightweight middleware designed to streamline and accelerate the process of updating model weights in LLM inference engines, critical for reinforcement learning and model updates.  [Read the original repo](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing Fast Weight Updates:** Efficiently updates model weights, with support for models like Kimi-K2 (1 trillion parameters) across thousands of GPUs in ~20 seconds.
*   **Two Update Methods:**
    *   **Broadcast:** Synchronous, optimized for speed when many inference instances need updates. Ideal for the default method.
    *   **P2P (Peer-to-Peer):**  Allows dynamic addition of new inference instances (e.g., restarts, scaling) while minimizing impact on existing workloads, leveraging `mooncake-transfer-engine`.
*   **Optimized Broadcast Implementation:**  A three-stage pipeline (H2D, broadcast, reload) that overlaps communication and data copy to maximize performance and reduce latency.
*   **Seamless Integration with vLLM:**  Provides straightforward integration with the popular vLLM inference engine.
*   **Flexible Installation:** Supports both broadcast and P2P update methods with easy-to-use `pip` installation.
*   **Reuse Weights from Existing Instances:** New instances can quickly obtain a copy of the checkpoint by setting `--load-metas-file global_metas.pkl`.
*   **FP8 Quantization Support:** Includes a patch for FP8 quantization in vLLM (tested on DeepSeek-V3.1 and Kimi-K2) to achieve further performance gains.

## Architecture

The core logic resides within the `ParameterServer` class, co-located with inference engines. It offers two primary implementations for weight updates:

*   **Broadcast:** The preferred and fastest method, ideal for synchronous updates across numerous inference instances.  See `_update_per_bucket`.
*   **P2P:** Designed for scenarios involving dynamically added inference instances. It utilizes the `mooncake-transfer-engine` for peer-to-peer weight transfer from existing instances' CPUs to the new instances' GPUs, mitigating disruption to ongoing workloads. See `_update_per_bucket_p2p`.

### Optimized Weight Broadcast

The *Broadcast* implementation utilizes a three-stage process for optimized data transfer:

1.  **H2D (Host-to-Device):** Transfer weights from CPU memory (potentially from disk or a training engine) to GPU memory.
2.  **Broadcast:** Distribute weights among checkpoint engine workers using CUDA IPC.
3.  **Reload:** Inference engines select the necessary subset of weights from the broadcasted data.

The system orchestrates this transfer using metadata and a pipelined architecture for maximized performance, as illustrated below.

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmark

Performance benchmarks were conducted using various models and hardware configurations.  See the table below for detailed results.

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   FP8 tests require vLLM patches (see [FP8 quantization](#fp8-quantization)).
*   The "Device Info" column describes device and parallelism setups (e.g., 256-GPU TP16 = 16 vLLM instances, each with 16-way tensor parallelism).
*   Bucket size (related to IPC) is also provided in the table, as it is related to the update duration.
*   P2P times were tested for updating no more than two nodes (16 GPUs) within the cluster.

## Installation

**Install the Fastest Broadcast Implementation:**

```bash
pip install checkpoint-engine
```

**Install the Flexible P2P Implementation (Includes `mooncake-transfer-engine`):**

```bash
pip install 'checkpoint-engine[p2p]'
```

**RDMA Configuration (Optional):**

If the `NCCL_IB_HCA` environment variable is set, the engine will automatically select network devices.  Otherwise, it will attempt to divide all RDMA devices among the ranks.

## Getting Started

**Prerequisites:** H800 or H20 machine with 8 GPUs and the latest vLLM, including the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit (available in the main branch).

1.  **Install vLLM:**

    ```bash
    cd /opt && git clone https://github.com/vllm-project/vllm && cd vllm
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    ```
2.  **Install Checkpoint Engine:**

    ```bash
    uv pip install 'checkpoint-engine[p2p]'
    ```
3.  **Download a Model (e.g., Qwen3-235B-A22B-Instruct-2507):**

    ```bash
    hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```
4.  **Start vLLM in Dev Mode (with dummy load format):**

    ```bash
    VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
        --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
        --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
        --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
    ```

5.  **Update Weights with Checkpoint Engine (concurrently):**

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

### Reuse Weights from Existing Instances

1.  **Start Existing Instances (to save metas):**
    ```bash
    torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
        --sleep-time 300 --save-metas-file global_metas.pkl
    ```
2.  **Start New Instances (load metas):**
    ```bash
    torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
    ```

### FP8 Quantization

FP8 quantization is not natively supported in vLLM when updating weights. A patch file (`patches/vllm_fp8.patch`) is provided and has been tested for DeepSeek-V3.1 and Kimi-K2, with a PR pending in the vLLM project.

### Test

Run a simple correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested only with vLLM, with potential for easy integration with other frameworks like SGLang.
*   The full three-stage pipeline (H2D, Broadcast, Reload) is not yet fully implemented.
*   The P2P update method can be further optimized.

## Acknowledgments

This project builds upon the vLLM interface (see [vLLM PR #24295](https://github.com/vllm-project/vllm/pull/24295)).  Thanks to [youkaichao](https://github.com/youkaichao) for valuable feedback and insights.