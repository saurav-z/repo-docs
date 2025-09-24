# Checkpoint Engine: Efficient Weight Updates for Large Language Models

Quickly update model weights in your LLM inference engines with the **Checkpoint Engine**, a lightweight and efficient solution for reinforcement learning and model fine-tuning. [View the original repository](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Fast Weight Updates:** Update a 1 Trillion parameter model (like Kimi-K2) across thousands of GPUs in approximately 20 seconds.
*   **Inplace Weight Updates:** Optimized for efficient and lightweight inplace weight updates.
*   **Broadcast and P2P Methods:** Supports both synchronous broadcast for optimal speed and P2P for dynamic instance addition.
*   **Optimized Weight Broadcast:**  Utilizes a 3-stage pipeline (H2D, broadcast, reload) to maximize performance.
*   **Integration with vLLM:** Seamlessly integrates with vLLM for inference, offering a flexible solution.
*   **FP8 Support:** Includes patches for FP8 quantization, enhancing performance.
*   **Reuse Weights:** Allows new instances to easily join existing ones and reuse their weights.

## Architecture

The Checkpoint Engine utilizes a `ParameterServer` class, co-located with inference engines, to handle weight updates. It offers two primary update methods:

*   **Broadcast:** The fastest method, ideal for synchronous updates across a large number of inference instances.  See `_update_per_bucket`.
*   **P2P (Peer-to-Peer):**  Enables updates for dynamically added inference instances without disrupting existing workloads, leveraging the [`mooncake-transfer-engine`](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#use-python-package) for RDMA transfers. See `_update_per_bucket_p2p`.

### Optimized Weight Broadcast Details

The Broadcast implementation efficiently transfers sharded weights from CPU memory to inference engine GPUs, which consists of three stages:

1.  **H2D (Host-to-Device):** Transfers weights to GPU memory.
2.  **Broadcast:** Distributes weights among checkpoint engine workers using a CUDA IPC buffer.
3.  **Reload:** The inference engine copies the required weights from the broadcasted data.

The engine orchestrates the entire process, including data transfer planning and control of the inference engine via ZeroMQ. To maximize performance, data transfers are organized in a pipelined manner, enabling overlapping communication and copy operations, described in [Kimi-K2 Technical Report](https://arxiv.org/abs/2507.20534).

## Benchmark

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   All results are tested by [`examples/update.py`](./examples/update.py) and use [vLLM v0.10.2rc1](https://github.com/vllm-project/vllm/tree/v0.10.2rc1) as inference engine.
*   FP8 test needs additional vLLM patches, see [FP8 quantization](#fp8-quantization).
*   Device Info: we tested various combinations of devices and parallelism setups. For example, a 256-GPU TP16 setup means that we deploy 16 vLLM instances, each with 16-way tensor parallelism.
*   Since update duration is related to IPC bucket size, we provide the bucket size in the table.
*   The P2P time were tested for updating no more than two nodes (16 GPUs) (`ParameterServer.update(ranks=range(0, 16))`) out of the entire cluster.

## Installation

Install the core functionality:

```bash
pip install checkpoint-engine
```

Install with P2P support (includes `mooncake-transfer-engine`):

```bash
pip install 'checkpoint-engine[p2p]'
```

*   If `NCCL_IB_HCA` is set, Checkpoint Engine uses it to select network devices. If not set, it reads all RDMA devices.

## Getting Started

Prerequisites:

*   An H800 or H20 machine with 8 GPUs.
*   Latest vLLM with the [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit.

Installation and setup:

```bash
cd /opt && git clone https://github.com/vllm-project/vllm && cd vllm
uv venv --python 3.12 --seed
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

Install Checkpoint Engine:

```bash
uv pip install 'checkpoint-engine[p2p]'
```

Download a test model (e.g., `Qwen/Qwen3-235B-A22B-Instruct-2507`):

```bash
hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

Start vLLM in dev mode:

```bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
    --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
    --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
```

Update weights using Checkpoint Engine:

```bash
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

### Reusing Weights from Existing Instances

1.  Start existing instances with `--save-metas-file global_metas.pkl` and a long `--sleep-time`.
2.  New instances can then obtain a copy of the checkpoint using `--load-metas-file global_metas.pkl`.

```bash
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
    --sleep-time 300 --save-metas-file global_metas.pkl
```

```bash
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

### FP8 Quantization

FP8 quantization requires a patch in [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch) for vLLM.  This is only tested with DeepSeek-V3.1 and Kimi-K2. A [PR](https://github.com/vllm-project/vllm/pull/24488) is opened to the vLLM project.

### Test

Run a simple correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested only with vLLM.
*   Full 3-stage pipeline not yet implemented.
*   P2P update method can be further optimized.

## Acknowledgments

This project is based on the vLLM interface from https://github.com/vllm-project/vllm/pull/24295. Special thanks to [youkaichao](https://github.com/youkaichao) for comments and insights.