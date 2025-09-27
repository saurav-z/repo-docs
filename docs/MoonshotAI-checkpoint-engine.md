# Checkpoint Engine: Accelerating LLM Weight Updates for Efficient Inference

**Checkpoint Engine** is a high-performance middleware designed to efficiently update model weights in large language model (LLM) inference engines, enabling rapid weight updates critical for reinforcement learning and model adaptation. For more details, visit the original repository: [MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine).

## Key Features

*   **Blazing-Fast Updates:**  Achieves rapid weight updates, for example, updating a 1 Trillion parameter model (Kimi-K2) in approximately 20 seconds across thousands of GPUs.
*   **Optimized Weight Broadcast:** Implements a 3-stage pipeline (H2D, broadcast, reload) for efficient data transfer and utilizes CUDA IPC buffers for shared memory access to ensure high-throughput updates across inference instances.
*   **P2P Weight Transfer:** Provides Peer-to-Peer (P2P) updates for dynamically added inference instances, leveraging `mooncake-transfer-engine` to minimize disruption to existing workloads.
*   **Flexible Deployment:** Compatible with various device and parallelism configurations, supporting BF16 and FP8 precision with vLLM, and easy integration with other frameworks.
*   **Seamless Integration:**  Provides installation via pip, simplifying integration into your existing LLM workflows.
*   **Weight Reuse:** Enables new checkpoint-engine instances to reuse weights from existing instances for efficient checkpoint loading.

## Architecture

Checkpoint Engine centers around the `ParameterServer` class, co-located with inference engines.  It offers two primary weight update methods:

*   **Broadcast:**  The fastest method, suitable for synchronous updates across numerous inference instances. This is the default setting.

*   **P2P:** Designed for dynamic instance additions, the P2P method utilizes the `mooncake-transfer-engine` for efficient weight transfer between existing instances and newly added instances, without affecting active workloads.

### Optimized Weight Broadcast

The broadcast implementation is optimized to efficiently transfer sharded weights from CPU memory to GPU memory using the following three-stage pipeline:

1.  **H2D:** Move weights to GPU memory (from disk or training engine).
2.  **Broadcast:** Distribute weights among checkpoint engine workers via CUDA IPC buffers.
3.  **Reload:** The inference engine selectively copies weights from the broadcasted data.

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmarks

The table below details performance benchmarks for different LLMs across various hardware configurations.

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

*   All tests use [vLLM v0.10.2rc1](https://github.com/vllm-project/vllm/tree/v0.10.2rc1) as the inference engine.
*   FP8 requires additional vLLM patches.
*   The bucket size for IPC data transfer is in the table.
*   P2P times are for updating a subset of nodes (16 GPUs).

## Installation

Install the optimized broadcast implementation:

```bash
pip install checkpoint-engine
```

To enable the flexible P2P implementation, install with the `p2p` extra:

```bash
pip install 'checkpoint-engine[p2p]'
```

## Getting Started

1.  **Prerequisites:** Requires an H800 or H20 machine with 8 GPUs and the latest vLLM.  Ensure you have the [/collective\_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit.

2.  **Set up vLLM:**

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

4.  **Download Model:**

```bash
hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

5.  **Run vLLM:** Start vLLM in dev mode with dummy load format and the custom worker extension:

```bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
    --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
    --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
```

6.  **Update Weights:**  Use the following command to update weights via the checkpoint engine:

```bash
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

### Reuse weights from existing instances

1.  **Save Metas:**  Start existing instances with `--save-metas-file global_metas.pkl` to save global metas.

```bash
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
    --sleep-time 300 --save-metas-file global_metas.pkl
```

2.  **Load Metas:**  New instances can obtain the checkpoint by setting `--load-metas-file global_metas.pkl`.

```bash
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

### FP8 quantization

For FP8 quantization, apply the patch in [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch).

## Test

Run a simple correctness test:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested primarily with vLLM; easy integration with other frameworks (SGLang) is planned.
*   The full 3-stage pipeline optimization, mentioned in our paper, is not yet fully implemented.
*   P2P implementation can be further optimized.

## Acknowledgments

This project utilizes the vLLM interface; thanks for the comments and insights from [youkaichao](https://github.com/youkaichao).