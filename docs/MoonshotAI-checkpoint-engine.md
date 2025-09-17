# Checkpoint Engine: Accelerating LLM Weight Updates for Fast Inference and Reinforcement Learning

**Checkpoint Engine enables rapid weight updates for large language models (LLMs), crucial for efficient inference and reinforcement learning, with updates for a 1 Trillion parameter model taking only 20 seconds.** Learn more about the power of Checkpoint Engine on its original repository: [MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Efficient Weight Updates:**  Designed for fast and reliable weight updates, optimizing for both synchronous (broadcast) and dynamic (P2P) scenarios.
*   **Two Update Methods:**
    *   **Broadcast:**  The fastest method for synchronous updates, ideal for large-scale, simultaneous weight changes across inference instances.
    *   **P2P (Peer-to-Peer):**  Enables weight updates for dynamically added inference instances without disrupting existing workloads, using [mooncake-transfer-engine](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#use-python-package).
*   **Optimized Broadcast Implementation:**  Employs a three-stage pipeline (H2D, broadcast, reload) to maximize performance, with fallback to serial execution when memory is constrained.
*   **Integration with vLLM:**  Seamlessly integrates with the vLLM inference engine, allowing quick updates to existing model weights.
*   **Flexible Deployment:** Supports both CPU and GPU-based deployments with RDMA transfer via `mooncake-transfer-engine` for flexible updating between existing and new instances.
*   **FP8 Quantization Support:** Offers experimental support for FP8 quantization with provided patches for vLLM.

## Architecture

The `ParameterServer` class forms the core of Checkpoint Engine, residing alongside inference engines and handling the weight update logic. It supports two main update implementations:

*   **Broadcast:**  The default and fastest method, optimized for synchronous weight updates across multiple inference instances.
*   **P2P:**  Facilitates weight updates for newly added instances, preventing disruption to existing workloads.

### Optimized Weight Broadcast

The Broadcast implementation uses a three-stage approach:

1.  **H2D:** Moves weights to GPU memory.
2.  **Broadcast:** Broadcasts data among checkpoint engine workers to shared CUDA IPC buffer.
3.  **Reload:** The inference engine chooses which weights to copy.

## Benchmarks

Checkpoint Engine demonstrates impressive performance with various LLM models and hardware configurations.  See benchmark results in the original README.

## Installation

Install the core package for the fastest broadcast implementation:

```bash
pip install checkpoint-engine
```

Install the P2P implementation with support for RDMA transfers:

```bash
pip install 'checkpoint-engine[p2p]'
```

## Getting Started

**Prerequisites:**  H800 or H20 machine with 8 GPUs and latest vLLM installed. Requires the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit from the vLLM main branch.

1.  **Clone and Install vLLM:**

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

4.  **Start vLLM:**

    ```bash
    VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
        --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
        --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
        --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
    ```

5.  **Update Weights using Checkpoint Engine:**

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

### Reusing Weights from Existing Instances

1.  Start existing instances with `--save-metas-file global_metas.pkl` and `--sleep-time 300`.
2.  New instances can then load checkpoint data using `--load-metas-file global_metas.pkl`.

## FP8 Quantization

Experimental support for FP8 quantization is available with patches in the `patches/vllm_fp8.patch` file. This patch is tested in DeepSeek-V3.1 and Kimi-K2.

## Testing

Run a correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested with vLLM only (but easy to integrate with other frameworks).
*   The three-stage pipeline is not fully implemented (potential optimization).
*   The P2P update method can be further optimized.

## Acknowledgments

This project leverages the vLLM interface, with thanks to [youkaichao](https://github.com/youkaichao) for insights and comments.