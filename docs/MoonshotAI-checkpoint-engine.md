# Checkpoint Engine: Efficient Weight Updates for Large Language Models

**Checkpoint Engine accelerates LLM inference by providing a fast and flexible middleware for updating model weights, essential for reinforcement learning and model refinement.**  Learn more about the original project here: [https://github.com/MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)

<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Key Features

*   **Blazing-Fast Weight Updates:** Achieve rapid weight updates, such as updating a 1-Trillion parameter model (like Kimi-K2) across thousands of GPUs in approximately 20 seconds.
*   **Optimized Broadcast Implementation:** Leverage a three-stage data transfer process (H2D, broadcast, reload) for efficient weight distribution within a cluster.
*   **Flexible P2P Implementation:** Supports dynamic instance additions and updates through the `mooncake-transfer-engine`, minimizing impact on existing workloads.
*   **Performance Benchmarks:**  Demonstrated efficiency with various models, including GLM-4.5-Air, Qwen3, DeepSeek-V3.1, and Kimi-K2.
*   **Seamless Integration with vLLM:** Designed for easy integration with vLLM and other inference engines, ensuring streamlined model updates.
*   **RDMA Support:**  Leverages RDMA for fast transfer between ranks.

## Architecture

The `Checkpoint Engine` centers around the `ParameterServer` class, co-located with the inference engines, which offers two core weight update methods:

*   **Broadcast:** Ideal for synchronous weight updates across a large number of inference instances.  This is the default and fastest update method.
*   **P2P (Peer-to-Peer):** Handles updates for dynamically added instances, such as those added during restarts or scaling.  This utilizes the `mooncake-transfer-engine` to transfer weights directly between instances.

### Optimized Weight Broadcast Details

The Broadcast implementation utilizes a three-stage pipeline to optimize weight transfer:

1.  **H2D:** Transfer weights to GPU memory, sourced from disk or the training engine.
2.  **Broadcast:** Distribute weights across checkpoint engine workers using a CUDA IPC buffer shared with the inference engine.
3.  **Reload:** The inference engine selects the necessary weights from the broadcast data.

Checkpoint Engine orchestrates this process by gathering metadata, creating a transfer plan, and controlling the inference engine via a ZeroMQ socket. The pipeline maximizes performance by overlapping communication and data copy operations (see the image below).

<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

## Benchmarks

Performance comparisons are available in the benchmark table provided in the original README.

## Installation

**Install the optimized broadcast implementation:**

```bash
pip install checkpoint-engine
```

**Install the flexible P2P implementation (which also installs `mooncake-transfer-engine`):**

```bash
pip install 'checkpoint-engine[p2p]'
```

## Getting Started

Detailed setup instructions, including requirements, are in the original README.

## Further Information

*   **FP8 Quantization:** Instructions for FP8 quantization are available in the original README.
*   **Testing:** Run a simple correctness test via `torchrun --nproc-per-node 8 tests/test_update.py`.
*   **Reusing weights from existing instances:** Instructions are available in the original README.

## Limitations and Future Work

*   Currently primarily tested with vLLM; however, it's designed for integration with other frameworks.
*   Potential for further pipeline optimization.
*   Optimization of the P2P update method.

## Acknowledgments

The project uses the vLLM interface, with thanks to [youkaichao](https://github.com/youkaichao) for comments and insights.