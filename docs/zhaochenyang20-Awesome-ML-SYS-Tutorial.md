# Awesome-ML-SYS-Tutorial: Your Guide to the World of ML Systems

Dive into the fascinating intersection of Machine Learning and Systems with this comprehensive repository of learning notes, code, and insights. Explore cutting-edge topics and practical implementations to build a strong foundation in ML Systems.  [View the original repository](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial).

**Key Features:**

*   **RLHF System Development Notes:** Comprehensive notes and code on Reinforcement Learning from Human Feedback (RLHF) systems, including analysis of training processes, weight update mechanisms, and optimization strategies.
    *   Analysis of multi-turn training time consumption and profile in verl.
    *   Deep dives into FSDP training backends, weight update mechanisms, and verl parameter walkthroughs.
    *   Exploration of multi-turn interaction and tool use with verl.
    *   Implementation of long context RLHF for industry-level projects.
    *   Development of rollout interfaces for complex RLHF systems.
    *   Includes deep analysis of existing frameworks like OpenRLHF, with focus on understanding computational flows and practical implementation.
*   **SGLang Learning Notes:**  Explore the capabilities of SGLang, a powerful framework for large language model (LLM) serving and inference.
    *   Constraint decoding and optimization techniques.
    *   In-depth code walkthroughs for SGLang engine and worker processes.
    *   Analysis of SGLang scheduling and routing mechanisms.
    *   Insights into VLLM integration with SGLang.
*   **Scheduling and Routing:** Understand the critical aspects of LLM serving efficiency.
    *   Explore prefill and decode strategies.
    *   ModelServer and front-end distribution systems based on SGLang.
*   **ML System Fundamentals:** Solidify your understanding of core ML system concepts.
    *   CUDA Graph analysis.
    *   NCCL and NVIDIA TOPO for efficient communication.
    *   PyTorch Distributed for parallel training.
    *   Model Quantization and Evaluation techniques (BF16, AWQ).
*   **Development Guides:** Practical guides for setting up and managing your ML system development environment.
    *   Docker usage for environment management.
    *   Setting up a clean and efficient development environment with `uv`.
    *   Special tokens and chat template understanding.
    *   CI/CD for documentation deployment.