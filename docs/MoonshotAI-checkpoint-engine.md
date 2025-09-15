# Checkpoint Engine: Efficient Model Weight Updates for LLM Inference

**Checkpoint Engine provides a fast and efficient solution for updating model weights in large language model (LLM) inference engines, critical for reinforcement learning and model updates.**  Explore the original repository at [https://github.com/MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Optimized Weight Update:** Implements efficient methods for updating model weights, including Broadcast and Peer-to-Peer (P2P) transfer.
*   **Broadcast Implementation:**  Leverages a three-stage pipeline (H2D, broadcast, reload) for rapid weight updates across inference instances.
*   **P2P Implementation:** Enables dynamic addition of inference instances and weight transfer using  `mooncake-transfer-engine`.
*   **Performance:**  Achieves fast update times, updating a 1 trillion parameter model in approximately 20 seconds across thousands of GPUs.
*   **Integration with vLLM:** Designed for seamless integration with vLLM and compatible with other frameworks like SGLang.
*   **FP8 Support:** Provides a patch for FP8 quantization support in vLLM (with specific model compatibility).

## Architecture

The `ParameterServer` class is at the core of Checkpoint Engine, collocated with inference engines. It offers two primary weight update methods:

*   **Broadcast:** Ideal for synchronous updates across a large number of inference instances. This is the fastest method and is the recommended default.  See `_update_per_bucket`.
*   **P2P (Peer-to-Peer):** Designed for scenarios where new instances are added dynamically. This method uses `mooncake-transfer-engine` to transfer weights between existing instances (CPUs) and new instances (GPUs). See `_update_per_bucket_p2p`.

### Optimized Weight Broadcast

The *Broadcast* implementation efficiently broadcasts sharded weights from CPU memory to inference instances. The process is optimized using a three-stage data transfer pipeline:

1.  **H2D:** Transferring weights to GPU memory.
2.  **Broadcast:**  Broadcasting data among checkpoint engine workers via CUDA IPC buffer shared with inference engine.
3.  **Reload:** Inference engine selects and copies required weights from the broadcast data.

Checkpoint Engine orchestrates the entire transfer, including planning and execution, using a ZeroMQ socket to control the inference engine. A pipeline with overlapped communication and copy maximizes performance. More details can be found in the [Kimi-K2 Technical Report](https://arxiv.org/abs/2507.20534).

## Benchmark

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   Results were tested using `examples/update.py` and [vLLM v0.10.2rc1](https://github.com/vllm-project/vllm/tree/v0.10.2rc1).
*   FP8 tests require specific vLLM patches (see [FP8 quantization](#fp8-quantization)).
*   Device Info describes the device and parallelism configuration (e.g., 256-GPU TP16 uses 16 vLLM instances with 16-way tensor parallelism).
*   Bucket size is provided as it affects update duration.
*   P2P times were tested for updates to a subset of the cluster.

## Installation

Install the core package:

```bash
pip install checkpoint-engine
```

Install with P2P support (which will install `mooncake-transfer-engine`):

```bash
pip install 'checkpoint-engine[p2p]'
```

If the `NCCL_IB_HCA` environment variable is set, checkpoint-engine will automatically select network devices. Otherwise, it will attempt to divide all RDMA devices among ranks.

## Getting Started

### Prerequisites

*   H800 or H20 machine with 8 GPUs
*   Latest vLLM (with [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit)

### Setup

1.  **Clone and install vLLM:**

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

3.  **Download a Model:**

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

### Reuse Weights from Existing Instances

1.  **Save Metas:** Start existing instances and save metas to a file. Use `--sleep-time 300` to keep instances alive.

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
        --sleep-time 300 --save-metas-file global_metas.pkl
    ```

2.  **Load Metas:** New instances can load the checkpoint by specifying the metas file:

    ```bash
    torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
    ```

### FP8 Quantization

For FP8 quantization, apply the patch provided in [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch). *Note: This patch is primarily tested with DeepSeek-V3.1 and Kimi-K2. Other models may have compatibility issues.* A PR for vLLM integration is pending review.

### Test

Run the correctness tests:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested with vLLM.
*   The ideal three-stage pipeline is not fully implemented.
*   The P2P update method is not fully optimized.

## Acknowledgments

This project leverages the vLLM interface from [https://github.com/vllm-project/vllm/pull/24295](https://github.com/vllm-project/vllm/pull/24295).  Thanks to [youkaichao](https://github.com/youkaichao) for insights and comments.