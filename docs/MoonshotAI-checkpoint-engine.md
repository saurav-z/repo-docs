# Checkpoint Engine: Efficient Weight Updates for LLM Inference

**Speed up Large Language Model (LLM) inference with Checkpoint Engine, a lightweight middleware designed for rapid in-place weight updates, crucial for reinforcement learning and dynamic model deployments.** [View the original repository](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing Fast Updates:** Achieve weight updates in LLMs with up to 1 trillion parameters in just ~20 seconds across thousands of GPUs.
*   **Optimized Broadcast:** Efficiently distribute updated weights using a three-stage pipeline (H2D, broadcast, reload) for high-performance data transfer.
*   **Flexible P2P Updates:** Seamlessly integrate new inference instances without disrupting existing workloads using a Peer-to-Peer (P2P) approach, leveraging `mooncake-transfer-engine`.
*   **VLLM Integration:** Designed for easy integration with vLLM and compatible with other frameworks.
*   **FP8 Support:** Includes patches for FP8 quantization in vLLM, enabling further optimization.
*   **Easy to Use:** Simple installation and straightforward setup for rapid deployment.

## Architecture

Checkpoint Engine employs a `ParameterServer` class, co-located with inference engines, offering two primary weight update methods:

*   **Broadcast:** Synchronously updates weights across a large number of instances for optimal speed.
    *   The implementation uses the following 3 stages:
        1.  H2D: Moving weights to GPU memory. These weights may come from disk or the training engine.
        2.  broadcast: broadcast among checkpoint engine workers; the data results in a CUDA IPC buffer shared with inference engine.
        3.  reload: inference engine decides what subset of weights to copy from the broadcasted data.
    *   See `_update_per_bucket`.
*   **P2P:** Allows dynamic addition of new instances without impacting existing ones. Utilizes `mooncake-transfer-engine` for efficient weight transfer.
    *   See `_update_per_bucket_p2p`.

### Optimized Weight Broadcast

The broadcast method uses a pipelined approach to maximize performance:

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmarks

The table below demonstrates performance across various models and hardware configurations:

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   **Note:** Performance depends on IPC bucket size. The bucket size is shown in the table above.

## Installation

Install using pip:

```bash
pip install checkpoint-engine
```

To include P2P functionality (which will install `mooncake-transfer-engine`):

```bash
pip install 'checkpoint-engine[p2p]'
```

If `NCCL_IB_HCA` is set, Checkpoint Engine will select network devices automatically. Otherwise, it reads all RDMA devices and tries to divide them between ranks.

## Getting Started

1.  **Prerequisites:**
    *   H800 or H20 machine with 8 GPUs
    *   Latest vLLM version. Includes [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit.
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

4.  **Download a Model (Example: Qwen3-235B):**

```bash
hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

5.  **Start vLLM (in dev mode):**

```bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
    --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
    --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
```

6.  **Update Weights:**

```bash
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

### Reuse Weights from Existing Instances

1.  **Save Metas:** Start existing instances with `--save-metas-file global_metas.pkl` and `--sleep-time 300`.
2.  **Load Metas:** New instances can then reuse weights by setting `--load-metas-file global_metas.pkl`.

### FP8 Quantization

Use the provided patch at `patches/vllm_fp8.patch` for correct FP8 weight updates.
This patch has been tested with DeepSeek-V3.1 and Kimi-K2.

### Test

Run a basic test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested primarily with vLLM.
*   The optimal three-stage pipeline is not fully implemented.
*   The P2P update method can be further optimized.

## Acknowledgments

This project utilizes the vLLM interface as implemented in https://github.com/vllm-project/vllm/pull/24295. Thank you to [youkaichao](https://github.com/youkaichao) for their insights.