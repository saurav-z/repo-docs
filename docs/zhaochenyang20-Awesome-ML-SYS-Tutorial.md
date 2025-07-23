# Awesome-ML-SYS-Tutorial: Your Comprehensive Guide to ML System Development

**Unlock the potential of Machine Learning Systems with this comprehensive repository, featuring learning notes, code, and insights for both beginners and experts.** ([Original Repository](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial))

This repository is a curated collection of learning notes, code, and reflections on the intersection of Machine Learning and Systems (ML SYS). It serves as a valuable resource for anyone interested in delving into this rapidly evolving field.

**Key Features:**

*   **RLHF System Development Notes:** Deep dives into Reinforcement Learning from Human Feedback (RLHF) systems, covering topics such as FSDP training, weight update mechanisms, and the intricacies of multi-turn interaction.
*   **SGLang Learning Notes:** Comprehensive exploration of SGLang, a powerful framework for language model serving, including code walkthroughs, scheduler analysis, and constraint decoding techniques.
*   **Scheduling and Routing:** Insights into efficient model serving, including prefill/decode optimization and model server design.
*   **ML System Fundamentals:** A solid grounding in core ML system concepts, covering distributed training with PyTorch, CUDA graphs, NCCL, and model quantization.
*   **Development Guides:** Practical guides for setting up development environments, including Docker and CI/CD pipelines, and understanding essential concepts like special tokens and chat templates.
*   **Multi-Language Support:** Many documents are available in both English and Simplified Chinese to accommodate a global audience.

## Table of Contents

### RLHF System 开发笔记
*   verl multi-turn training 时间消耗
*   RL 系统深思：FSDP 训练后端
*   RL 系统深思：深入理解权重更新机制
*   verl 参数速览
*   深入浅出理解 verl 源码（Rollout）
*   深入浅出理解 verl 源码（初始化）
*   从 tokenizer 视角来分析 Agentic 多轮训练的复杂性
*   Search-R1 & veRL-SGLang: Train LLMs with Multi-Turn RL to Reason and Call a Search Engine
*   SGLang, verl, OpenBMB 与清华大学团队联合开源：在主流 RLHF 框架上首次支持多轮交互与工具调用
*   Kimi K1.5: Long Context RL 的成功实践
*   SGLang-veRL Server：从 Engine 到 Server，我们需要更灵活的 RLHF rollout 接口
*   Rule-based Reward
*   HybridFlow veRL 原文浅析
*   扩展 OpenRLHF 的推理引擎
*   SWE-Bench：如何构造 LLM 时代的优秀 Benchmark
*   浅析以 OpenRLHF 为代表的 post-training 系统的计算流程
*   图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读
*   Latency optimization for weight updates
*   浅析主流 Alignment 算法与 NeMo-Aligner 框架

### SGLang 学习笔记
*   查看 HuggingFace 模型结构
*   Constraint Decoding 的概念、方法与优化
*   SGLang Code Walk Through
*   Walk Through SGLang / VLLM Worker
*   Walk Through SGLang Scheduler
*   Latency Accelerate For Weight Updates
*   Reward / Embed Model Sever Engine 现状浅析
*   SGLang 后端原文解析
*   小白视角：利用 vllm serve 新的 Embedding Model
*   小白视角：利用 SGL 来 Serve Embedding Model
*   小白视角：vllm 迁移到 SGLang 的体验与收获

### Scheduling and Routing
*   Mooncake：将 P / D 分离进行到底
*   prefill 和 decode 该分离到不同的卡上么？
*   基于 chunked prefill 理解 prefill 和 decode 的计算特性
*   ModelServer：基于 SGLang 的前端分发系统

### ML System 基本功
*   基于 torch-memory-savor 浅析 CUDA Graph
*   NCCL 与 NVIDIA TOPO
*   PyTorch Distributed
*   Give me BF16 or Give Me Death，当下量化方法的全面评测
*   AWQ：模型量化应当关注激活值
*   [原创][深度][PyTorch] DDP系列第一篇：入门教程
*   nvidia-smi命令详解和一些高阶技巧介绍

### 开发指南
*   How to use docker
*   配置清爽的开发环境
*   一文理解 special tokens 和 chat template
*   在 CI 上编译 jupyter notebook 并部署为文档

### 未公开部分
*   NCCL and SGLang (Eng)