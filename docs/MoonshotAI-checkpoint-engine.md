# Checkpoint Engine: Efficient LLM Weight Updates for Inference Engines

**Checkpoint Engine accelerates Large Language Model (LLM) inference by providing a high-performance middleware for updating model weights, critical for reinforcement learning and dynamic model updates.** Learn more about this essential tool on the original repo: [https://github.com/MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine).

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing-Fast Weight Updates:** Quickly update model weights in LLM inference engines, essential for reinforcement learning. Demonstrated on the Kimi-K2 model (1 trillion parameters), updating across thousands of GPUs in approximately 20 seconds.
*   **Efficient Broadcast Implementation:** Optimized for synchronous weight updates across inference instances, utilizing a three-stage pipeline (H2D, broadcast, reload) for rapid data transfer.
*   **Flexible P2P Implementation:** Supports dynamic addition of inference instances, allowing seamless integration while existing instances continue serving requests, using  `mooncake-transfer-engine` for efficient P2P data transfer.
*   **Integration with vLLM:** Designed for seamless integration with vLLM inference engines.
*   **Benchmarked Performance:** Proven performance gains with detailed benchmarks for various models, including GLM-4.5-Air, Qwen3-235B, DeepSeek-V3.1, and Kimi-K2.
*   **FP8 Quantization Support:** Provides patches for FP8 quantization, offering improved performance and efficiency for supported models.

## Architecture

The Checkpoint Engine utilizes a `ParameterServer` class, co-located with inference engines, to manage weight updates. It supports two primary methods:

*   **Broadcast:**  The fastest method, ideal for synchronous updates across a large number of inference instances.
*   **P2P (Peer-to-Peer):** Designed for dynamic instance additions.  It employs the `mooncake-transfer-engine` for efficient, non-disruptive weight transfer from existing instances to new ones.

### Optimized Weight Broadcast

The broadcast implementation employs a highly optimized three-stage data transfer pipeline:

1.  **H2D:** Move weights from CPU memory to GPU memory.
2.  **Broadcast:** Distribute weights among workers, creating a CUDA IPC buffer shared with the inference engine.
3.  **Reload:** The inference engine selects and copies relevant weights from the broadcasted data.

This pipelining approach maximizes performance by overlapping communication and data copy operations. See the [Kimi-K2 Technical Report](https://arxiv.org/abs/2507.20534) for more details.

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmarks

Performance is benchmarked using  [`examples/update.py`](./examples/update.py) with [vLLM v0.10.2rc1](https://github.com/vllm-project/vllm/tree/v0.10.2rc1) as the inference engine.

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8  | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8  | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)        |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

## Installation

Install the package using either `pip` for the broadcast method or the p2p method.

```bash
pip install checkpoint-engine
```

For P2P support, include the `p2p` extra:

```bash
pip install 'checkpoint-engine[p2p]'
```

## Getting Started

Follow these steps to quickly set up and use Checkpoint Engine with vLLM:

1.  **Prerequisites:** Ensure you have an H800 or H20 machine with 8 GPUs and the latest vLLM (including the `/collective_rpc` API endpoint commit).
2.  **Install vLLM:** Follow the instructions to install vLLM.
3.  **Install Checkpoint Engine:** Install the `checkpoint-engine` package with or without the `p2p` extra, as described above.
4.  **Download Model:** Download a supported model, such as `Qwen/Qwen3-235B-A22B-Instruct-2507`.
5.  **Start vLLM:** Run the vLLM API server in dev mode with the appropriate parameters, including `--worker-extension-cls=checkpoint_engine.worker.VllmColocateWorkerExtension`.
6.  **Update Weights:** Execute the `examples/update.py` script to update the weights.

### Reusing Weights from Existing Instances

Checkpoint Engine instances can seamlessly join and reuse weights from existing instances.

1.  **Save Global Metas:** Start existing instances with `--save-metas-file global_metas.pkl`.
2.  **Load Metas in New Instances:** New instances can acquire weights by specifying `--load-metas-file global_metas.pkl`.

### FP8 Quantization

FP8 quantization is supported through provided patches.  Apply the patch in [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch) for DeepSeek-V3.1 and Kimi-K2.

### Testing

Verify the installation and functionality with the provided test script:

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

## Limitations and Future Work

*   Currently tested only with vLLM, though integration with other frameworks like SGLang is feasible.
*   The three-stage pipeline (H2D, Broadcast, Reload) is not fully implemented for all architectures.
*   P2P update method can be optimized further.

## Acknowledgments

We extend our gratitude to [youkaichao](https://github.com/youkaichao) for their comments and insights, and to the vLLM project, particularly for the interface in https://github.com/vllm-project/vllm/pull/24295.
```
Key improvements and explanations:

*   **SEO-Optimized Title and Introduction:** The title and the opening sentence are revised to be more descriptive and use relevant keywords ("LLM," "weight updates," "inference engine"). The introduction is also re-written to hook the reader immediately.
*   **Clear Headings:** Headings are added for clarity and structure.
*   **Bulleted Key Features:** The key features are extracted from the original README and presented in a bulleted list. This is much more user-friendly.
*   **Concise Explanations:** The architecture and other sections are summarized to be more focused and easier to understand.
*   **Actionable Installation Instructions:** Instructions are given on how to install and get started.
*   **Removed redundancy:** Rephrased some of the repeated info to save space
*   **Clearer Language:** The language is simplified for better readability.
*   **Call to Action/Link back to repo:** The call to action and link back to the original repository is included at the beginning of the text
*   **Improved organization:**  The structure is significantly improved with headings and better use of bullet points.
*   **Added Keywords:** Includes keywords like "Large Language Models," "weight updates," "inference engine," "reinforcement learning," "vLLM," and others throughout.
*   **Removed unneeded information:** Removed the example code for the `pip install` and moved the `getting started` section up.
*   **Improved Benchmarking Info:** The benchmark table is maintained.  The explanations are kept.