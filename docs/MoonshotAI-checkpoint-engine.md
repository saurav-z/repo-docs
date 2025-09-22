# Checkpoint Engine: Efficient Weight Updates for LLM Inference

**Checkpoint Engine accelerates Large Language Model (LLM) inference by providing a fast and efficient middleware for updating model weights, crucial for reinforcement learning and dynamic model updates.** [View the original repository](https://github.com/MoonshotAI/checkpoint-engine)

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing-Fast Weight Updates:** Implements optimized weight updates, enabling rapid model adjustments for large-scale LLMs.
*   **Two Update Methods:**  Offers both **Broadcast** (for synchronous updates across many instances) and **P2P** (for dynamically adding instances without disrupting existing workloads) update strategies.
*   **Optimized Broadcast Implementation:** Utilizes a three-stage data transfer pipeline (H2D, broadcast, reload) with overlapped communication and copy to maximize performance.
*   **Scalable Design:**  Tested and benchmarked with models up to 1 Trillion parameters across thousands of GPUs.
*   **Seamless Integration with vLLM:** Provides easy integration with vLLM and other LLM inference engines.
*   **Flexible Installation:**  Supports both pip installation of the core engine and optional P2P functionality via `mooncake-transfer-engine`.

## Architecture

Checkpoint Engine's core logic resides in the `ParameterServer` class, which is co-located with the inference engines. It provides two main update methods:

*   **Broadcast:** Designed for synchronous weight updates across a large number of inference instances. This is the default and fastest method.
*   **P2P (Peer-to-Peer):** Enables dynamic addition of new inference instances without impacting existing ones. It leverages `mooncake-transfer-engine` for efficient data transfer between CPUs and GPUs.

### Optimized Weight Broadcast

The *Broadcast* implementation is optimized for speed. It involves these three stages:

1.  **H2D:** Moving weights to GPU memory.
2.  **Broadcast:** Broadcast among checkpoint engine workers, resulting in a CUDA IPC buffer shared with the inference engine.
3.  **Reload:** The inference engine selects and copies the necessary weights from the broadcasted data.

This process is orchestrated by the checkpoint engine, utilizing metadata to create a plan and controlling the inference engine via a ZeroMQ socket.  The data transfers are organized into a pipeline, enabling overlapped communication and copy for maximum performance (see the `pipeline.png` in the original README).  If memory is insufficient, the engine will automatically fall back to serial execution.

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmarks

The following table presents benchmark results, which show the time taken for weight updates using both Broadcast and P2P methods, for various LLM models on different hardware configurations.

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   FP8 tests require additional vLLM patches (see [FP8 Quantization](#fp8-quantization)).
*   Device Info:  These configurations demonstrate the flexibility of the engine.  For example, a "256-GPU TP16" setup means 16 vLLM instances, each using 16-way tensor parallelism.
*   The bucket size is included in the table since update duration is related to IPC bucket size.
*   The P2P times were tested by updating no more than two nodes (16 GPUs) (`ParameterServer.update(ranks=range(0, 16))`) out of the entire cluster.

## Installation

Install the core package:

```bash
pip install checkpoint-engine
```

Install with P2P support:

```bash
pip install 'checkpoint-engine[p2p]'
```

Set `NCCL_IB_HCA` to automatically select network devices for different ranks. If not set, the system will identify and attempt to divide RDMA devices among ranks.

## Getting Started

This section provides detailed instructions for running checkpoint-engine with vLLM, including setting up the environment, downloading a model, and running an update example.

1.  **Prerequisites:**  Requires an H800 or H20 machine with 8 GPUs and the latest vLLM, including the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit (available in the main branch).
2.  **Set up vLLM:** Clone, and install vLLM:

```bash
cd /opt && git clone https://github.com/vllm-project/vllm && cd vllm
uv venv --python 3.12 --seed
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

3.  **Install checkpoint-engine:**

```bash
uv pip install 'checkpoint-engine[p2p]'
```

4.  **Download a Test Model:**  The example uses `Qwen/Qwen3-235B-A22B-Instruct-2507` (BF16):

```bash
hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

5.  **Start vLLM:** Start vLLM in dev mode with `--load-format dummy`, and set `--worker-extension-cls=checkpoint_engine.worker.VllmColocateWorkerExtension`:

```bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
    --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
    --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
```

6.  **Run Checkpoint Engine Update:**  Use `torchrun` to update weights. You do not need to wait for vLLM to get ready:

```bash
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

### Reuse Weights from Existing Instances

You can configure new checkpoint-engine instances to reuse weights from existing ones:

1.  **Save Metas:**  Start existing instances with `--save-metas-file global_metas.pkl` and `--sleep-time 300`.

```bash
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
    --sleep-time 300 --save-metas-file global_metas.pkl
```

2.  **Load Metas:** New instances can then load weights by setting `--load-metas-file global_metas.pkl`.

```bash
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

### FP8 Quantization

FP8 quantization support currently requires a patch in `patches/vllm_fp8.patch` to properly update weights.  This patch has been tested with DeepSeek-V3.1 and Kimi-K2.

A PR is currently opened to the vLLM project.

### Testing

Run a simple correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   **Framework Support:** Primarily tested with vLLM but designed for easy integration with other frameworks.
*   **Pipeline Optimization:**  The complete three-stage pipeline described in the paper is not yet fully implemented.
*   **P2P Optimization:** The P2P update method could be improved for increased efficiency.

## Acknowledgments

This project utilizes the same vLLM interface as the one proposed in https://github.com/vllm-project/vllm/pull/24295. Thanks for the valuable comments and insights from [youkaichao](https://github.com/youkaichao).