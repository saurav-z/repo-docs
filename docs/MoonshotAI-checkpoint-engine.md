# Checkpoint Engine: Fast and Efficient Weight Updates for LLM Inference

Update your large language model weights quickly and efficiently with Checkpoint Engine, a lightweight middleware designed for seamless integration with LLM inference engines like vLLM. [Learn more and contribute on GitHub](https://github.com/MoonshotAI/checkpoint-engine).

<!-- Image of the checkpoint-engine architecture -->
<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing Fast Updates:** Update 1 Trillion parameter models in seconds.
*   **Inplace Weight Updates:** Efficiently updates model weights directly within the inference engine.
*   **Two Update Methods:**
    *   **Broadcast:** Synchronous, high-speed updates ideal for large-scale deployments.
    *   **P2P:** Peer-to-peer updates, enabling dynamic scaling and updates without disrupting existing inference workloads.
*   **Optimized Broadcast Implementation:** Three-stage data transfer pipeline for maximum performance.
*   **Seamless vLLM Integration:** Easy to integrate with vLLM and other frameworks.
*   **Benchmarked Performance:** Proven performance on models like GLM-4.5-Air, Qwen3, DeepSeek-V3.1, and Kimi-K2.

## Architecture

Checkpoint Engine's core functionality resides within the `ParameterServer` class, co-located with your inference engines.  It provides two primary methods for weight updates:

*   **Broadcast:** The default and fastest method, ideal for synchronous weight updates across many inference instances. This implementation leverages the `_update_per_bucket` function.
*   **P2P:** Designed for scenarios where new inference instances are added dynamically, ensuring no disruption to ongoing workloads.  Uses the [`mooncake-transfer-engine`](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#use-python-package) to transfer weights between instances via P2P communication, utilizing the `_update_per_bucket_p2p` function.

### Optimized Weight Broadcast Explained

The *Broadcast* implementation efficiently handles the transfer of sharded weights from CPU memory to a cluster of inference instances, often with different sharding patterns. It uses a three-stage process:

1.  **H2D:** Transfers weights to GPU memory.
2.  **Broadcast:** Broadcasts the weights among checkpoint engine workers, creating a shared CUDA IPC buffer for the inference engine.
3.  **Reload:** The inference engine determines the specific subset of weights to copy from the broadcasted data.

Checkpoint-engine orchestrates the transfer process, creating a plan based on metadata and deciding appropriate bucket sizes for data transfer. It controls the inference engine through a ZeroMQ socket and organizes the data transfers into a pipeline for overlapped communication and copy operations. See [Kimi-K2 Technical Report](https://arxiv.org/abs/2507.20534) for details.

<!-- Image of the pipeline -->
<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

*Note: Pipelining requires more GPU memory.  If insufficient memory is available, Checkpoint Engine will fall back to serial execution.*

## Benchmarks

Performance benchmarks comparing update times across different models and hardware configurations. Bucket size is also provided.

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*All results above are tested by [`examples/update.py`](./examples/update.py) using [vLLM v0.10.2rc1](https://github.com/vllm-project/vllm/tree/v0.10.2rc1).*

*Notes:*

*   FP8 testing requires additional vLLM patches; see [FP8 Quantization](#fp8-quantization).
*   Device Info provides information on tested configurations, including device type and parallelism setups (e.g., 256-GPU TP16 setup means 16 vLLM instances, each with 16-way tensor parallelism).
*   The P2P update times were tested for updating up to two nodes (16 GPUs).

## Installation

Install with either the broadcast or P2P implementation:

**Install the Fastest Broadcast Implementation:**

```bash
pip install checkpoint-engine
```

**Install with Flexible P2P Implementation:**

```bash
pip install 'checkpoint-engine[p2p]'
```

*   Setting `NCCL_IB_HCA` enables automatic selection of network devices for each rank. If not set, all RDMA devices are read, and an attempt is made to divide them among the ranks.

## Getting Started

### Prerequisites

*   H800 or H20 machine with 8 GPUs.
*   Latest vLLM
*   Include the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit, available on the vLLM main branch.

1.  **Set up vLLM:**

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
    Start vLLM in dev mode, setting `--load-format dummy` and `--worker-extension-cls=checkpoint_engine.worker.VllmColocateWorkerExtension`:

    ```bash
    VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
        --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
        --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
        --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
    ```

5.  **Update weights using Checkpoint Engine:**

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
    ```

### Reuse Weights from Existing Instances

New Checkpoint Engine instances can join existing instances, reusing their weights.

1.  **Save Global Metas:** Start existing instances with `--save-metas-file global_metas.pkl` and `--sleep-time 300`:

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
        --sleep-time 300 --save-metas-file global_metas.pkl
    ```

2.  **Load Global Metas:** New instances can obtain a copy of the checkpoint by setting `--load-metas-file global_metas.pkl`.

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
    ```

### FP8 Quantization

FP8 quantization requires a patch: [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch)

*This patch has been tested with DeepSeek-V3.1 and Kimi-K2. Other models may encounter compatibility issues.*

A PR has been opened to the vLLM project, [PR](https://github.com/vllm-project/vllm/pull/24488).

### Testing

Run a simple correctness test

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested only with vLLM; easy to integrate with frameworks such as SGLang.
*   The three-stage pipeline, mentioned in the paper, is not yet fully implemented.
*   The P2P method is currently not optimized and can be improved.

## Acknowledgments

This project uses the same vLLM interface in https://github.com/vllm-project/vllm/pull/24295 . Thanks for the comments and insights from [youkaichao](https://github.com/youkaichao).