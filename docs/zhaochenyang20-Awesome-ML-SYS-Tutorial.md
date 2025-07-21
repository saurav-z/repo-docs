# Awesome-ML-SYS-Tutorial: Your Guide to the Cutting Edge of Machine Learning Systems

This repository offers comprehensive learning notes, code, and insights into the exciting intersection of Machine Learning and Systems (ML-SYS).  [Explore the original repository on GitHub](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial).

**Key Features:**

*   **In-Depth RLHF System Development Notes:**
    *   Analysis of verl multi-turn training time consumption.
    *   Understanding FSDP training backends.
    *   Deep dives into weight update mechanisms.
    *   Code walkthroughs of verl, including rollout and initialization.
    *   Exploration of multi-turn tokenization challenges.
    *   Integration of the Search-R1 framework and support for multi-turn interaction and tool usage.
    *   Analysis of Kimi K1.5's Long Context RL implementation.
    *   Rollout server based on veRL
    *   Post-training systems and the calculation flow of RLHF.
    *   Latency optimization for weight updates.
    *   Analysis of Alignment algorithms and NeMo-Aligner framework.
*   **SGLang Learning Notes:**
    *   Understanding of Constraint Decoding.
    *   Comprehensive SGLang code walkthrough.
    *   Worker
    *   Scheduler
    *   Latency Accelerate For Weight Updates
    *   Reward / Embed Model Sever Engine
    *   Analysis of SGLang backend
    *   Using vllm serve and SGLang to serve Embedding Models
*   **Scheduling and Routing:**
    *   Analysis of Mooncake and prefill vs. decode.
    *   Model Server based on SGLang
*   **ML System Fundamentals:**
    *   CUDA Graph and CUDA Graph implementation.
    *   NCCL and NVIDIA TOPO.
    *   PyTorch Distributed communication practices.
    *   Model quantization (BF16, AWQ).
    *   Details of DDP and all_reduce.
*   **Development Guides:**
    *   Docker usage for development environment management.
    *   Configuration of a clean development environment.
    *   Understanding special tokens and chat templates in transformers.
    *   CI compilation of Jupyter notebooks for documentation.

**Explore the cutting edge of ML-SYS!**