# Checkpoint Engine: Efficient Weight Updates for LLM Inference

**Checkpoint Engine** accelerates the crucial process of updating model weights in large language model (LLM) inference engines, enabling rapid updates for reinforcement learning and other dynamic applications.  Learn more about this powerful tool on its [original repository](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features:

*   **Fast Weight Updates:** Efficiently updates model weights, enabling rapid iteration and deployment.  Updates a 1 Trillion parameter model in approximately 20 seconds across thousands of GPUs.
*   **Optimized Broadcast Implementation:** Utilizes a three-stage pipeline (H2D, Broadcast, Reload) for optimized weight transfer, minimizing latency and maximizing performance.
*   **Flexible P2P Updates:** Supports P2P weight updates for dynamic inference instance additions, ensuring minimal disruption to existing workloads.  Leverages `mooncake-transfer-engine` for RDMA transfer between different ranks.
*   **Integration with vLLM:** Designed for seamless integration with vLLM, enhancing its capabilities for dynamic model updates.
*   **Benchmark Results:** Proven performance gains demonstrated on various models with different device configurations, with comprehensive benchmark results provided.
*   **Easy Installation:** Provides straightforward installation options via pip, including options for both broadcast and P2P functionalities.
*   **FP8 Quantization Support:** Includes guidance and patches for supporting FP8 quantization in vLLM.
*   **Example Usage:** Includes clear instructions and examples on how to use the library for both initial setup and reuse of weights from existing instances.

## Architecture:

Checkpoint Engine employs a `ParameterServer` class located alongside inference engines, providing two main update methods:

*   **Broadcast:** Ideal for synchronous weight updates across numerous inference instances.  This is the default and fastest implementation.
*   **P2P (Peer-to-Peer):** Enables weight updates for dynamically added instances, preserving existing workloads. Utilizes `mooncake-transfer-engine` for efficient data transfer.

### Optimized Weight Broadcast Details:

The Broadcast implementation optimizes data transfer in three stages:

1.  **H2D:** Transfer weights to GPU memory.
2.  **Broadcast:** Broadcast among checkpoint engine workers using a CUDA IPC buffer shared with the inference engine.
3.  **Reload:** The inference engine selects and loads relevant weights from the broadcasted data.

The entire transfer process is orchestrated to create a plan, including determining the appropriate bucket size for data transfer. Data transfers are organized into a pipeline with communication and copy operations overlapping to maximize performance, as illustrated below:

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmark Performance:

The library's performance is validated with comprehensive benchmarks.  The results, including both Broadcast and P2P update times, are shown below:

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

## Installation:

Install using pip for the fastest broadcast implementation:

```bash
pip install checkpoint-engine
```

To utilize the P2P implementation, which includes `mooncake-transfer-engine` for RDMA transfers:

```bash
pip install 'checkpoint-engine[p2p]'
```

## Getting Started:

Before beginning, ensure you have an H800 or H20 machine with 8 GPUs and the latest vLLM version. Include the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit.

1.  **Set up vLLM:**

    ```bash
    cd /opt && git clone https://github.com/vllm-project/vllm && cd vllm
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    ```

2.  **Install checkpoint-engine:**

    ```bash
    uv pip install 'checkpoint-engine[p2p]'
    ```

3.  **Download a Model (Example):**

    ```bash
    hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

4.  **Start vLLM:**

    ```bash
    VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
        --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
        --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
        --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
    ```

5.  **Update Weights:**

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

### Reuse weights from existing instances:

1.  **Start Existing Instances:** Save global metas file:

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
        --sleep-time 300 --save-metas-file global_metas.pkl
    ```

2.  **Start New Instances:** Load global metas file:

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
    ```

### FP8 Quantization

For FP8 support, apply the patch at [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch) and refer to the [PR](https://github.com/vllm-project/vllm/pull/24488) for related discussions.

### Test

Run the following to test Checkpoint Engine functionality:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work:

*   Currently, only tested with vLLM, but easily adaptable to other frameworks.
*   Potential for further optimization of the three-stage pipeline.
*   P2P update method optimization.

## Acknowledgments:

This project builds upon the vLLM interface as seen in https://github.com/vllm-project/vllm/pull/24295.  Thanks to [youkaichao](https://github.com/youkaichao) for their insights and feedback.