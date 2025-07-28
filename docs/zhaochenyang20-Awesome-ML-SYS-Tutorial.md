# Awesome-ML-SYS-Tutorial: Your Guide to Machine Learning Systems

Are you interested in the intersection of Machine Learning and Systems? This repository offers comprehensive notes, code, and insights into the exciting world of ML systems, covering RLHF, SGLang, and more, offering practical knowledge for both beginners and experienced practitioners. Check out the [original repo](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)!

## Key Features:

*   **RLHF System Development:** In-depth analysis of RLHF systems, including code walkthroughs, system design considerations, and optimization techniques.
*   **SGLang Deep Dive:** Comprehensive notes on SGLang, a domain-specific language for large language models, including code walkthroughs and practical applications.
*   **Scheduling and Routing:** Explore techniques for optimizing model serving, including prefill/decode separation and model server design.
*   **ML System Fundamentals:** Essential knowledge of distributed training, CUDA, and model quantization.
*   **Development Guides:** Learn best practices for setting up development environments, using Docker, and understanding special tokens in transformers.
*   **Multilingual Support:** Access content in both English and Simplified Chinese, fostering a wider community.

## Table of Contents:

### RLHF System 开发笔记
*   [Systematic Analysis of verl Multi-Turn Training Time Consumption](rlhf/verl/multi-turn/tool_examples/profile.md)
*   [RL System Deep Dive: FSDP Training Backend](rlhf/sys-design/readme-2.md)
*   [RL System Deep Dive: Understanding Weight Update Mechanisms](rlhf/sys-design/readme-1.md)
*   [verl Parameter Overview](rlhf/verl/multi-turn/code-walk-through/readme-5.md)
*   [In-Depth Understanding of verl Source Code (Rollout)](rlhf/verl/multi-turn/code-walk-through/readme-2.md)
*   [In-Depth Understanding of verl Source Code (Initialization)](rlhf/verl/multi-turn/code-walk-through/readme.md)
*   [Analyzing Agentic Multi-Turn Training Complexity from a Tokenizer Perspective](rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking_ZH.md)
*   [Search-R1 & veRL-SGLang: Train LLMs with Multi-Turn RL to Reason and Call a Search Engine](rlhf/verl/multi-turn//tool_examples/verl-multiturn-searchR1-like_ZH.md)
*   [SGLang, verl, OpenBMB and Tsinghua University Team Jointly Open Source: First Support for Multi-Turn Interaction and Tool Calling on Mainstream RLHF Frameworks](rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md)
*   [Kimi K1.5: Successful Implementation of Long Context RL](rlhf/partial-rollout/readme.md)
*   [SGLang-veRL Server: From Engine to Server, We Need a More Flexible RLHF Rollout Interface](rlhf/verl/server-based/veRL-server-based-rollout.md)
*   [Rule-based Reward](https://zhuanlan.zhihu.com/p/13211508979)
*   [HybridFlow veRL Original Text Analysis](rlhf/verl/readme.md)
*   [Extending the Inference Engine of OpenRLHF](rlhf/OpenRLHF/develop-log.md)
*   [SWE-Bench: How to Construct an Excellent Benchmark for the LLM Era](https://zhuanlan.zhihu.com/p/16292266518)
*   [Analyzing the Calculation Process of Post-Training Systems Represented by OpenRLHF](rlhf/OpenRLHF/readme.md)
*   [Illustrated Large Model RLHF Series: PPO Principles and Source Code Interpretation that Everyone Can Understand](https://zhuanlan.zhihu.com/p/677607581) and [Illustrated Distributed Training Process Based on Ray in OpenRLHF](https://zhuanlan.zhihu.com/p/12871616401)
*   [Latency optimization for weight updates](sglang/latency-accelerte-for-weight-updates/readme.md)
*   [Analyzing Mainstream Alignment Algorithms and the NeMo-Aligner Framework](https://zhuanlan.zhihu.com/p/5220718268)

### SGLang 学习笔记
*   [View HuggingFace Model Structure](https://zhuanlan.zhihu.com/p/9912733791)
*   [Constraint Decoding Concepts, Methods and Optimization](sglang/constraint-decoding/readme.md)
*   [SGLang Code Walk Through](sglang/code-walk-through/readme.md)
*   [Walk Through SGLang / VLLM Worker](sglang/sglang-worker/readme.md)
*   [Walk Through SGLang Scheduler](sglang/sglang-scheduler/readme-CN.md)
*   [Latency Accelerate For Weight Updates](sglang/latency-accelerte-for-weight-updates/readme-CN.md)
*   [Reward / Embed Model Sever Engine Status Analysis](https://zhuanlan.zhihu.com/p/4148050391)
*   [SGLang Backend Original Text Analysis](https://zhuanlan.zhihu.com/p/716543182)
*   [Beginner's Perspective: Utilizing vllm to Serve New Embedding Models](https://zhuanlan.zhihu.com/p/715857723)
*   [Beginner's Perspective: Utilizing SGL to Serve Embedding Models](https://zhuanlan.zhihu.com/p/715805386)
*   [Beginner's Perspective: Experience and Gain from Migrating vllm to SGLang](https://zhuanlan.zhihu.com/p/714833359)

### Scheduling and Routing
*   [Mooncake: Separating P / D to the End](https://zhuanlan.zhihu.com/p/1711346141)
*   [Should prefill and decode be separated onto different cards?](https://zhuanlan.zhihu.com/p/1280567902)
*   [Understanding the Computational Characteristics of Prefill and Decode Based on Chunked Prefill](https://zhuanlan.zhihu.com/p/718715866)
*   [ModelServer: A Frontend Distribution System Based on SGLang](https://zhuanlan.zhihu.com/p/718015016)

### ML System 基本功
*   [Analyzing CUDA Graph Based on torch-memory-savor](distributed/cuda-graph/readme.md)
*   [NCCL and NVIDIA TOPO](distributed/nccl/readme.md)
*   [PyTorch Distributed](distributed/torch-distributed/readme.md)
*   [Give me BF16 or Give Me Death, Comprehensive Evaluation of Current Quantization Methods](https://zhuanlan.zhihu.com/p/5485556270)
*   [AWQ: Model Quantization Should Focus on Activation Values](https://zhuanlan.zhihu.com/p/942485319)
*   [[Original][Deep][PyTorch] DDP Series Part 1: Introductory Tutorial](https://zhuanlan.zhihu.com/p/178402798)
*   [nvidia-smi Command Explanation and Introduction of Some Advanced Techniques](https://www.yourmetaverse.cn/deep_learning/199/)

### 开发指南
*   [How to use docker](engineer/how-to-use-docker/readme.md)
*   [Configure a Clean Development Environment](engineer/uv/readme.md)
*   [Understand Special Tokens and Chat Templates in One Article](transformers/special_tokens/special_tokens.md)
*   [Compile jupyter notebook on CI and deploy as documentation](https://zhuanlan.zhihu.com/p/2382351079)

### 未公开部分
*   [NCCL and SGLang](distributed/nccl/readme_en.md)