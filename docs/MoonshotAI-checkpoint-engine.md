# Checkpoint Engine: Efficient Weight Updates for LLM Inference

**Checkpoint Engine is a fast and lightweight middleware designed to efficiently update model weights in Large Language Model (LLM) inference engines, such as those used in reinforcement learning, and can update a 1 Trillion parameter model in under 20 seconds.**  For more details, visit the original repository: [MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine).

## Key Features

*   **Blazing-Fast Weight Updates:** Achieve rapid weight updates, crucial for dynamic model adjustments.
*   **Inplace Weight Update:** Implements efficient inplace weight updates.
*   **Two Update Methods:** Choose between Broadcast and P2P for optimal performance based on your needs.
*   **Optimized Broadcast:** Implements a pipelined data transfer for high performance.
*   **Seamless Integration:** Designed to work with leading inference engines like vLLM.
*   **Flexible Deployment:** Supports both single and multi-GPU setups, including large-scale deployments.
*   **Easy Installation:** Simple pip install for both broadcast and P2P functionalities.

## Architecture

Checkpoint Engine's core logic resides within the `ParameterServer` class, co-located with the inference engines. It offers two primary weight update implementations:

*   **Broadcast:** (Default) Synchronously updates weights across many inference instances, making it the fastest method. See `_update_per_bucket`.
*   **P2P (Peer-to-Peer):**  Dynamically adds new inference instances (e.g., due to restarts) without disrupting existing workloads using the `mooncake-transfer-engine`. See `_update_per_bucket_p2p`.

### Optimized Weight Broadcast

The Broadcast implementation is optimized to efficiently broadcast sharded weights from CPU memory to inference instances with potentially different sharding patterns. This process involves:

1.  **H2D:** Transferring weights to GPU memory.
2.  **Broadcast:** Distributing weights among checkpoint engine workers using CUDA IPC.
3.  **Reload:**  Inference engine selectively copies weights from broadcasted data.

A pipeline is used to maximize performance.

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmark

The following benchmarks demonstrate the speed of the Checkpoint Engine:

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   Tests were performed using [`examples/update.py`](./examples/update.py) and vLLM v0.10.2rc1.
*   FP8 tests require specific vLLM patches (see [FP8 Quantization](#fp8-quantization)).
*   Device Info specifies hardware and parallelism.
*   Bucket size (in GiB) is also listed in the table, as update duration is related to this.
*   P2P testing updated no more than two nodes (16 GPUs).

## Installation

Install the Checkpoint Engine with your preferred method:

*   **Fast Broadcast Implementation:**

    ```bash
    pip install checkpoint-engine
    ```

*   **Flexible P2P Implementation:**

    ```bash
    pip install 'checkpoint-engine[p2p]'
    ```

    Note: This installs `mooncake-transfer-engine` for RDMA transfer between ranks.
    Setting `NCCL_IB_HCA` allows automatic selection of network devices.

## Getting Started

Follow these steps to get up and running:

1.  **Hardware & Software:** Prepare an H800 or H20 machine with 8 GPUs and the latest vLLM (including [/collective\_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit).
2.  **Install vLLM:**

    ```bash
    cd /opt && git clone https://github.com/vllm-project/vllm && cd vllm
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    ```

3.  **Install Checkpoint Engine:**

    ```bash
    uv pip install 'checkpoint-engine[p2p]'
    ```

4.  **Download a Model:** We use `Qwen/Qwen3-235B-A22B-Instruct-2507` (BF16) as an example:

    ```bash
    hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

5.  **Start vLLM:**  Run vLLM in dev mode, setting `--load-format dummy` and using `--worker-extension-cls=checkpoint_engine.worker.VllmColocateWorkerExtension`:

    ```bash
    VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
        --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
        --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
        --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
    ```

6.  **Update Weights:** Run the update script.  No need to wait for vLLM to be ready:

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

### Reuse weights from existing instances

New checkpoint-engine instances can join existing instances and reuse their weights. This is simple to achieve.

First, start the existing instances with `--save-metas-file global_metas.pkl` to save global metas to a file and use `--sleep-time 300` to make sure they stay alive.

```bash
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
    --sleep-time 300 --save-metas-file global_metas.pkl
```

After a checkpoint is registered, new instances can obtain a copy of the checkpoint by setting `--load-metas-file global_metas.pkl`.

```bash
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

### FP8 Quantization

FP8 quantization requires modifications to vLLM for proper weight updates. A patch file ([`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch)) is provided. Currently, this is tested only with DeepSeek-V3.1 and Kimi-K2.

A [PR](https://github.com/vllm-project/vllm/pull/24488) is opened to the vLLM project.

### Test

Run a simple correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested primarily with vLLM.
*   The full three-stage pipeline optimization mentioned in the paper is not yet implemented.
*   The P2P update method can be further optimized.

## Acknowledgments

Thanks to the vLLM project for the shared interface in https://github.com/vllm-project/vllm/pull/24295 . Thanks for the comments and insights from [youkaichao](https://github.com/youkaichao).