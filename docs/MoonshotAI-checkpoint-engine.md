# Checkpoint Engine: Efficient LLM Weight Updates for Faster Inference

**Checkpoint Engine** is a high-performance middleware designed to swiftly update model weights in Large Language Model (LLM) inference engines, crucial for reinforcement learning and model updates.  It can update a 1 Trillion parameter model in just 20 seconds. For more details, visit the original repository: [https://github.com/MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing-Fast Weight Updates:**  Leverages optimized broadcast and P2P communication methods for rapid weight propagation.
*   **Inplace Weight Updates:**  Implements efficient inplace weight updates, minimizing memory overhead.
*   **Two Update Methods**:
    *   **Broadcast:** Synchronous updates for large-scale synchronous updates.
    *   **P2P:** Enables dynamic instance additions, supporting seamless weight updates for new instances without affecting existing workloads.
*   **Optimized Broadcast Implementation:** A three-stage data transfer pipeline (H2D, broadcast, reload) for maximum performance.
*   **VLLM Compatibility:**  Designed for seamless integration with vLLM and easily adaptable to other inference frameworks.
*   **Flexible Deployment:** Supports both CPU-based and GPU-accelerated weight transfer.

## Architecture

The core logic resides in the `ParameterServer` class, which offers two main implementations:

*   **Broadcast:** This method is optimized for synchronous weight updates across numerous inference instances. It's the default and fastest option.
*   **P2P (Peer-to-Peer):**  This approach is ideal for dynamically adding new inference instances. It utilizes the `mooncake-transfer-engine` for direct GPU-to-GPU weight transfer between existing and new instances, ensuring minimal disruption.

### Optimized Weight Broadcast

The *Broadcast* implementation efficiently handles weight updates in several stages:
1.  **H2D:** Transfers weights to GPU memory, possibly from disk or a training engine.
2.  **Broadcast:** Shares the data across checkpoint engine workers, using a CUDA IPC buffer for communication with the inference engine.
3.  **Reload:** The inference engine chooses which weights to copy from the broadcasted data.

The transfer is organized into a pipeline with overlapped communication and copy. Further details are in the [Kimi-K2 Technical Report](https://arxiv.org/abs/2507.20534).

## Benchmarks

| Model                                | Device Info   | Gather Metas | Update (Broadcast) | Update (P2P)             |
| :----------------------------------- | :------------ | :----------- | :------------------- | :----------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8   | 0.17s        | 3.94s (1.42GiB)     | 8.83s (4.77GiB)          |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8   | 0.46s        | 6.75s (2.69GiB)     | 16.47s (4.05GiB)         |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s        | 12.22s (2.38GiB)    | 25.77s (3.61GiB)         |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s        | 15.45s (2.93GiB)    | 36.24s (4.46GiB)         |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s        | 13.88s (2.54GiB)    | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s        | 21.50s (2.99GiB)    | 34.49s (4.57 GiB) |

*All results were tested using [`examples/update.py`](./examples/update.py) and vLLM v0.10.2rc1.*

## Installation

**For the fastest broadcast implementation:**

```bash
pip install checkpoint-engine
```

**For flexible P2P implementation (includes `mooncake-transfer-engine`):**

```bash
pip install 'checkpoint-engine[p2p]'
```

## Getting Started

### Prerequisites
*   An H800 or H20 machine with 8 GPUs.
*   Latest vLLM.
*   Ensure the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit from vLLM is included.

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

4.  **Start vLLM in Development Mode:**

```bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
    --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
    --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
```

5.  **Update Weights Using Checkpoint Engine (in a separate terminal):**

```bash
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

### Reuse Weights from Existing Instances

1.  **Start Existing Instances (saving metas):**

```bash
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
    --sleep-time 300 --save-metas-file global_metas.pkl
```

2.  **Start New Instances (loading metas):**

```bash
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

### FP8 Quantization

FP8 quantization is currently not natively supported in vLLM for weight updates. Apply the patch located at [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch) to enable this functionality.

### Test

Run a correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently primarily tested with vLLM, but adaptable to other frameworks.
*   The perfect three-stage pipeline (mentioned in the paper) is not yet fully implemented.
*   P2P update optimization can be improved.

## Acknowledgments

This project builds upon the vLLM interface, referencing [vllm-project/vllm/pull/24295](https://github.com/vllm-project/vllm/pull/24295), and appreciates the contributions from [youkaichao](https://github.com/youkaichao).