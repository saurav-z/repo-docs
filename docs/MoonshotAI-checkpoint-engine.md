# Checkpoint Engine: Accelerating LLM Weight Updates for Efficient Inference

**Checkpoint Engine** provides a high-performance middleware solution for updating model weights in Large Language Model (LLM) inference engines, crucial for reinforcement learning and model refinement. Get the code and contribute on [GitHub](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing Fast Weight Updates:**  Update a 1-trillion parameter model (like Kimi-K2) across thousands of GPUs in approximately 20 seconds.
*   **Two Update Methods:**
    *   **Broadcast:**  Optimized for synchronous updates across large numbers of inference instances. This is the default and fastest method.
    *   **P2P (Peer-to-Peer):** Enables dynamic updates for new inference instances, minimizing impact on existing workloads using [mooncake-transfer-engine](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#use-python-package).
*   **Optimized Broadcast Pipeline:**  Employs a three-stage pipelined data transfer process (H2D, broadcast, reload) to maximize performance and overlap communication and copy operations.
*   **Flexible Deployment:** Supports various device and parallelism setups, ensuring compatibility across different hardware configurations.
*   **Easy Integration:** Designed to be easily integrated with LLM inference frameworks, currently tested with vLLM.

## Architecture

Checkpoint Engine's core weight update logic resides in the `ParameterServer` class, which is co-located with the inference engines.  It offers two distinct update implementations: Broadcast and P2P.

### Optimized Weight Broadcast

The Broadcast implementation efficiently broadcasts sharded weights from CPU memory to a cluster of inference instances. The data transfer is organized into a three-stage pipeline:

1.  **H2D:** Transfers weights from CPU memory (or disk) to GPU memory.
2.  **Broadcast:** Shares data among checkpoint engine workers, resulting in a CUDA IPC buffer shared with the inference engine.
3.  **Reload:**  The inference engine selects which weights to copy from the broadcasted data.

This pipeline, illustrated below, optimizes for performance by overlapping communication and copy operations.

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

Pipelining requires additional GPU memory; checkpoint-engine can fall back to serial execution if the memory is not sufficient.

## Performance Benchmarks

The table below provides benchmark results for various models, demonstrating the performance gains with Checkpoint Engine.

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   **Notes:**  Results were tested using [`examples/update.py`](./examples/update.py) with [vLLM v0.10.2rc1](https://github.com/vllm-project/vllm/tree/v0.10.2rc1) as the inference engine.  FP8 testing requires specific vLLM patches (see [FP8 quantization](#fp8-quantization)).  The table includes bucket size used during the update process. P2P results are for updating up to two nodes.

## Installation

Install the broadcast (default) implementation:

```bash
pip install checkpoint-engine
```

Install the P2P implementation (requires `mooncake-transfer-engine`):

```bash
pip install 'checkpoint-engine[p2p]'
```

If you set `NCCL_IB_HCA` env, Checkpoint-engine will use it to auto select net devices for different ranks. If not set, it will read all RDMA devices and try to divide them into each rank.

## Getting Started

1.  **Prerequisites:** An H800 or H20 machine with 8 GPUs and the latest vLLM is required.  Ensure the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) from vLLM's main branch is included.

2.  **vLLM Setup:**

    ```bash
    cd /opt && git clone https://github.com/vllm-project/vllm && cd vllm
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    ```

3.  **Checkpoint Engine Installation:**

    ```bash
    uv pip install 'checkpoint-engine[p2p]'
    ```

4.  **Model Download:** Example model - `Qwen/Qwen3-235B-A22B-Instruct-2507`

    ```bash
    hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

5.  **Start vLLM (Dev Mode):**

    ```bash
    VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
        --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
        --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
        --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
    ```

6.  **Update Weights (Checkpoint Engine):**  Run this command simultaneously with vLLM.

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

### Reuse Weights from Existing Instances

1.  **Save Metas:** Start existing instances with `--save-metas-file global_metas.pkl` and `--sleep-time 300`.

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
        --sleep-time 300 --save-metas-file global_metas.pkl
    ```

2.  **Load Metas:** New instances can load checkpoints by setting `--load-metas-file global_metas.pkl`.

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
    ```

### FP8 Quantization

FP8 quantization in vLLM requires specific patches, provided in [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch). This patch is currently tested only on DeepSeek-V3.1 and Kimi-K2.  A PR is pending in the vLLM project to address this natively.

### Test

Run a correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   **Framework Dependency:** Primarily tested with vLLM, but easily adaptable to other frameworks.
*   **Pipeline Optimization:** Full implementation of the three-stage pipeline could further improve performance, particularly where H2D and broadcast do not conflict.
*   **P2P Optimization:**  Improve the P2P method to enable parallel data reception.

## Acknowledgments

This project builds upon the vLLM interface, specifically leveraging contributions from [youkaichao](https://github.com/youkaichao).  We appreciate the insights and collaboration.