# Awesome-ML-SYS-Tutorial: Your Guide to the Cutting Edge of Machine Learning Systems

**Dive into the fascinating world of Machine Learning Systems (ML SYS) with this comprehensive repository of learning notes, code, and insights.** This repository is a curated collection of resources for understanding and implementing cutting-edge ML SYS concepts. [Explore the original repository on GitHub](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) to deepen your understanding.

## Key Features:

*   **In-depth Learning Notes:** Comprehensive notes on ML SYS topics, providing insights into the core concepts.
*   **Code Examples and Walkthroughs:** Hands-on examples and code walkthroughs to help you understand and implement ML SYS techniques.
*   **RLHF System Development:** Detailed exploration of Reinforcement Learning from Human Feedback (RLHF) systems, including training, optimization, and implementation details.
*   **SGLang Deep Dive:** In-depth notes on SGLang, covering key concepts, code walkthroughs, and implementation details.
*   **Scheduling and Routing:** Resources on scheduling and routing techniques for efficient model serving.
*   **ML System Fundamentals:** Essential knowledge on distributed computing, quantization, and optimization for high-performance ML systems.
*   **Development Guides:** Practical guides on setting up development environments, using Docker, and understanding special tokens.
*   **Regular Updates:** The repository is continuously updated with new content and insights, ensuring you stay up-to-date with the latest advancements in ML SYS.

## Content Breakdown:

### RLHF System Development Notes
Dive into the inner workings of RLHF systems, from training optimizations to practical implementations.

*   [Systematic Analysis of verl multi-turn training](./rlhf/verl/multi-turn/tool_examples/profile.md)
*   [RL System Deep Dive: FSDP Training Backend](./rlhf/sys-design/readme-2.md)
*   [RL System Deep Dive: Understanding Weight Update Mechanisms](./rlhf/sys-design/readme-1.md)
*   [verl Parameters Overview](./rlhf/verl/multi-turn/code-walk-through/readme-5.md)
*   [Deep Dive into verl Source Code (Rollout)](./rlhf/verl/multi-turn/code-walk-through/readme-2.md)
*   [Deep Dive into verl Source Code (Initialization)](./rlhf/verl/multi-turn/code-walk-through/readme.md)
*   [Analyzing Agentic Multi-Turn Training Complexity from a Tokenizer Perspective](rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking_ZH.md)
*   [Integrating Search-R1 into verl-sglang Ecosystem](rlhf/verl/multi-turn//tool_examples/verl-multiturn-searchR1-like_ZH.md)
*   [Multi-Turn Interaction and Tool Calling Support in Mainstream RLHF Frameworks](rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md)
*   [Kimi K1.5: Long Context RL Implementation](./rlhf/partial-rollout/readme.md)
*   [SGLang-veRL Server: Flexible RLHF Rollout Interface](rlhf/verl/server-based/veRL-server-based-rollout.md)
*   [Rule-based Reward](https://zhuanlan.zhihu.com/p/13211508979)
*   [HybridFlow veRL Analysis](./rlhf/verl/readme.md)
*   [Extending OpenRLHF Inference Engine](./rlhf/OpenRLHF/develop-log.md)
*   [SWE-Bench: Constructing a Good Benchmark for the LLM Era](https://zhuanlan.zhihu.com/p/16292266518)
*   [Analysis of Post-Training Systems with OpenRLHF as an Example](./rlhf/OpenRLHF/readme.md)
*   [Diagram of Large Model RLHF Series: PPO Principles and Source Code Interpretation](https://zhuanlan.zhihu.com/p/677607581)
*   [Diagram of OpenRLHF Distributed Training Process based on Ray](https://zhuanlan.zhihu.com/p/12871616401)
*   [Latency Optimization for Weight Updates](./sglang/latency-accelerte-for-weight-updates/readme.md)
*   [Analysis of Mainstream Alignment Algorithms and NeMo-Aligner Framework](https://zhuanlan.zhihu.com/p/5220718268)

### SGLang Learning Notes
Gain expertise in SGLang, a powerful framework for large language models.

*   [Viewing HuggingFace Model Structure](https://zhuanlan.zhihu.com/p/9912733791)
*   [Constraint Decoding: Concepts, Methods and Optimization](./sglang/constraint-decoding/readme.md)
*   [SGLang Code Walk Through](./sglang/code-walk-through/readme.md)
*   [Walk Through SGLang / VLLM Worker](./sglang/sglang-worker/readme.md)
*   [Walk Through SGLang Scheduler](./sglang/sglang-scheduler/readme-CN.md)
*   [Latency Acceleration For Weight Updates](./sglang/latency-accelerte-for-weight-updates/readme-CN.md)
*   [Reward / Embed Model Server Engine Current State Analysis](https://zhuanlan.zhihu.com/p/4148050391)
*   [SGLang Backend Source Code Analysis](https://zhuanlan.zhihu.com/p/716543182)
*   [Beginner's Perspective: Using VLLM to Serve a New Embedding Model](https://zhuanlan.zhihu.com/p/715857723)
*   [Beginner's Perspective: Using SGL to Serve Embedding Model](https://zhuanlan.zhihu.com/p/715805386)
*   [Beginner's Perspective: Experience and Gains from Migrating VLLM to SGLang](https://zhuanlan.zhihu.com/p/714833359)

### Scheduling and Routing
Explore advanced scheduling and routing techniques for efficient model serving.

*   [Mooncake: Separating Prefill and Decode](https://zhuanlan.zhihu.com/p/1711346141)
*   [Should Prefill and Decode be Separated onto Different Cards?](https://zhuanlan.zhihu.com/p/1280567902)
*   [Understanding Prefill and Decode Computation Characteristics Based on Chunked Prefill](https://zhuanlan.zhihu.com/p/718715866)
*   [ModelServer: Front-End Distribution System Based on SGLang](https://zhuanlan.zhihu.com/p/718015016)

### ML System Fundamentals
Build a strong foundation in ML system essentials.

*   [CUDA Graph Analysis based on torch-memory-savor](./distributed/cuda-graph/readme.md)
*   [NCCL and NVIDIA TOPO](./distributed/nccl/readme.md)
*   [PyTorch Distributed](./distributed/torch-distributed/readme.md)
*   [Give me BF16 or Give Me Death, Comprehensive Evaluation of Current Quantization Methods](https://zhuanlan.zhihu.com/p/5485556270)
*   [AWQ: Model Quantization Should Focus on Activations](https://zhuanlan.zhihu.com/p/942485319)
*   [[Original][Deep][PyTorch] DDP Series Part 1: Beginner's Tutorial](https://zhuanlan.zhihu.com/p/178402798)
*   [Detailed Explanation of nvidia-smi Commands and Introduction of Some Advanced Techniques](https://www.yourmetaverse.cn/deep_learning/199/)

### Development Guides
Learn best practices for setting up and managing your ML SYS projects.

*   [How to use Docker](./engineer/how-to-use-docker/readme.md)
*   [Configuring a Clean Development Environment](./engineer/uv/readme.md)
*   [Understanding Special Tokens and Chat Templates](./transformers/special_tokens/special_tokens.md)
*   [Compiling Jupyter Notebooks on CI and Deploying as Documentation](https://zhuanlan.zhihu.com/p/2382351079)

### Unreleased Sections

*   [NCCL and SGLang](./distributed/nccl/readme_en.md)