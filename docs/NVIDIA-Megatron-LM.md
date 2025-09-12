<div align="center">

# Megatron-LM: Train Transformer Models at Scale with GPU-Optimized Performance

**Unlock the power of large language models with NVIDIA's Megatron-LM, a GPU-optimized library for training transformer models at scale.** ([Original Repo](https://github.com/NVIDIA/Megatron-LM))

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/Megatron-Core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.12.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

</div>

## Key Features

*   **High-Performance Training:** Optimized for NVIDIA GPUs, enabling efficient training of large-scale transformer models.
*   **Advanced Parallelism:** Supports Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP), Context Parallelism (CP), and Expert Parallelism (EP) for scaling model training.
*   **Composable Architecture:** Offers a modular and composable library (Megatron Core) for building custom training frameworks.
*   **Mixed Precision Training:** Supports FP16, BF16, and FP8 for faster training and reduced memory usage.
*   **Ecosystem Integration:** Compatible with popular libraries like Hugging Face Accelerate, Colossal-AI, and DeepSpeed, and integrated with libraries such as Megatron Energon, Transformer Engine, and NVIDIA's Resiliency Extension (NVRx).
*   **Production-Ready Examples:** Provides pre-configured training scripts and examples for popular models like GPT, LLaMA, DeepSeek, and Qwen.

## Quick Start

```bash
# 1. Install Megatron Core with required dependencies
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]

# 2. Clone repository for examples
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
```

**→ [Complete Installation Guide](#installation)** - Docker, pip variants (dev,lts,etc.), source installation, and system requirements

## What's New

*   🔄 **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Convert between Hugging Face and Megatron checkpoints.
*   🗺️ **[MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Future MoE features including DeepSeek-V3, Qwen3, and Blackwell optimizations.
*   🚀 **[GPT-OSS Implementation](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Incorporating features like YaRN RoPE scaling.
*   **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Optimized configs for MoE models such as DeepSeek-V3, Mixtral, and Qwen3.
*   **[Blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)** - New capabilities for multi-data center LLM training.

<details>
<summary>Previous News</summary>

- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)). 
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs. Explore the [Megatron Core intro](#Megatron Core) for more details.

</details>

<details>
<summary>Table of Contents</summary>

**Getting Started**
- [Quick Start](#-quick-start)
- [What's New](#whats-new)
- [Megatron Overview](#megatron-overview)
  - [Project Structure](#project-structure)
  - [Megatron-LM: Reference Implementation](#megatron-lm-reference-implementation)
  - [Megatron Core: Production Library](#megatron-core-production-library)
- [Installation](#installation) 
  - [Docker (Recommended)](#-docker-recommended)
  - [Pip Installation](#-pip-installation)
  - [Source Installation](#-source-installation)
  - [System Requirements](#system-requirements)

**Core Features**
- [Performance Benchmarking](#performance-benchmarking)
  - [Weak Scaling Results](#weak-scaling-results)
  - [Strong Scaling Results](#strong-scaling-results)
- [Ecosystem Libraries](#ecosystem-libraries)

**Training**
- [Training](#training)
  - [Getting Started](#getting-started)
  - [Data Preparation](#data-preparation)
- [Parallelism Strategies](#parallelism-strategies)
  - [Data Parallelism (DP)](#data-parallelism-dp)
  - [Tensor Parallelism (TP)](#tensor-parallelism-tp)
  - [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
  - [Context Parallelism (CP)](#context-parallelism-cp)
  - [Expert Parallelism (EP)](#expert-parallelism-ep)
  - [Parallelism Selection Guide](#parallelism-selection-guide)
- [Performance Optimizations](#performance-optimizations)

**Resources**
- [Examples](./examples/) - Training scripts and tutorials
- [Documentation](https://docs.nvidia.com/Megatron-Core/) - Official docs
- [Roadmaps](#roadmaps) - Development roadmaps and feature tracking
- [Community & Support](#-community--support) - Get help and contribute
  - [Getting Help](#getting-help)
  - [Contributing](#contributing)
  - [Citation](#citation)

</details>

## Megatron Overview

### Project Structure

```
Megatron-LM/
├── megatron/                    
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── inference/               # Inference server
│   ├── legacy/                  # Legacy components
│   └── post_training/           # Post-training (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

### Megatron-LM: Reference Implementation

The reference implementation provides everything needed to train state-of-the-art models.

**Best for:**

*   Training foundation models at scale.
*   Researching new architectures and techniques.
*   Learning distributed training concepts.
*   Quick experimentation.

**What you get:**

*   Pre-configured training scripts (GPT, Llama, DeepSeek, Qwen, etc.).
*   End-to-end examples from data prep to evaluation.
*   Research-focused tools.

### Megatron Core: Production Library

Megatron Core provides a composable library with GPU-optimized building blocks.

**Best for:**

*   Framework developers.
*   Research requiring custom training loops.
*   ML engineers building fault-tolerant pipelines.

**What you get:**

*   Composable transformer building blocks.
*   Advanced parallelism strategies (TP, PP, DP, EP, CP).
*   Pipeline schedules and distributed optimizers.
*   Mixed precision support (FP16, BF16, FP8).
*   GPU-optimized kernels.
*   High-performance dataloaders.
*   Model architectures (LLaMA, Qwen, GPT, Mixtral, Mamba, etc.)

## Ecosystem Libraries

**Libraries used by Megatron Core:**

*   **[Megatron Energon](https://github.com/NVIDIA/Megatron-Energon)** 📣 **NEW!** - Multi-modal data loader
*   **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)** - Optimized kernels and FP8 mixed precision support
*   **[Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext)** - Fault tolerant training

**Libraries using Megatron Core:**

*   **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Hugging Face ↔ Megatron checkpoint conversion
*   **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)** - Scalable toolkit for reinforcement learning.
*   **[NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)** - Enterprise framework.
*   **[TensorRT Model Optimizer (ModelOpt)](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** - Model optimization toolkit.

**Compatible with:** [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## Installation

### 🐳 Docker (Recommended)

Use the previous PyTorch NGC Container releases for best compatibility.

```bash
docker run --runtime --nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### Pip Installation

```bash
# Install the latest release with minimal dependencies (no Transformer Engine)
pip install megatron-core[dev]
```

```bash
# Install packages for LTS support NGC PyTorch 24.01
pip install megatron-core[lts]
```

For a version of Megatron Core with only torch, run:

```bash
pip install megatron-core
```

For dependencies required by Megatron-LM, please run:

```bash
pip install megatron-core[mlm]
```

### Source Installation

```bash
# Clone and install
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Optional: checkout specific release
git checkout core_r0.13.0

bash docker/common/install.sh --environment {dev,lts}
```

### System Requirements

#### Hardware Requirements

*   FP8 Support: NVIDIA Hopper, Ada, Blackwell GPUs
*   Recommended: NVIDIA Turing architecture or later

#### Software Requirements

*   CUDA/cuDNN/NCCL: Latest stable versions
*   PyTorch: Latest stable version
*   Transformer Engine: Latest stable version
*   Python: 3.12 recommended

## Performance Benchmarking

For performance results, see [NVIDIA NeMo Framework Performance Summary](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

Megatron-LM trains models from 2B to 462B parameters, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

*   Vocabulary size: 131,072 tokens
*   Sequence length: 4096 tokens
*   Model scaling: Varied hidden size, attention heads, and layers.
*   Communication optimizations: DP, TP, and PP

**Key Results:**

*   6144 H100 GPUs: Benchmarked 462B parameter model training
*   Superlinear scaling: MFU increases with model size
*   End-to-end measurement: Includes all operations
*   Production ready: Full training pipeline with checkpointing and fault tolerance
*   *Note: Performance results measured without training to convergence*

### Weak Scaling Results

![Weak scaling](images/weak_scaling.png)

### Strong Scaling Results

![Strong scaling](images/strong_scaling.png)

## Training

### Getting Started

```bash
# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

```bash
# LLama-3 Training Example
./examples/llama/train_llama3_8b_fp8.sh
```

### Data Preparation

### JSONL Data Format

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
```

### Basic Preprocessing

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

### Key Arguments

*   `--input`: Input file path.
*   `--output-prefix`: Output file prefix.
*   `--tokenizer-type`: Tokenizer type.
*   `--tokenizer-model`: Tokenizer model file path.
*   `--workers`: Parallel workers.
*   `--append-eod`: Append end-of-document token.

<!-- **→ [Complete Data Preparation Guide](./docs/data-preparation.md)** - Comprehensive guide covering advanced preprocessing, dataset collection, deduplication, and optimization strategies -->

## Parallelism Strategies

### Data Parallelism (DP)

#### Standard Data Parallel

```bash
# Standard DDP - replicate model on each GPU
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

#### Fully Sharded Data Parallel (FSDP)

```bash
--use-custom-fsdp

--use-torch-fsdp2

--data-parallel-sharding-strategy optim              # Shard optimizer states (ZeRO-1)
--data-parallel-sharding-strategy optim_grads        # Shard gradients + optimizer (ZeRO-2)
--data-parallel-sharding-strategy optim_grads_params # Shard parameters + gradients + optimizer (ZeRO-3)
```

### Tensor Parallelism (TP)

```bash
--tensor-model-parallel-size 4  # 4-way tensor parallelism
--sequence-parallel             # Enable sequence parallelism (recommended with TP)
```

### Pipeline Parallelism (PP)

```bash
--pipeline-model-parallel-size 8     # 8 pipeline stages
--virtual-pipeline-model-parallel-size 4  # Virtual pipeline for better load balancing
```

### Context Parallelism (CP)

```bash
--context-parallel-size 2                    # 2-way context parallelism
--cp-comm-type p2p                          # Communication: p2p, a2a, allgather, a2a+p2p
--hierarchical-context-parallel-sizes 2 4   # Hierarchical context parallelism
```

### Expert Parallelism (EP)

```bash
--expert-model-parallel-size 4  # 4-way expert parallelism
--num-experts 8                 # 8 experts per MoE layer
--moe-grouped-gemm              # Optimize expert computation
```

### Combining Parallelism Strategies

### Parallelism Selection Guide

Based on [NVIDIA NeMo production configurations](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance/recommended_model_configs):

| Model | Size | GPUs | TP | PP | CP | EP | Notes |
|-------|------|------|----|----|----|----|-------|
| **LLama-3** | 8B | 8 | 1 | 1 | 2 | 1 | CP for long seqlen (8K) |
| **LLama-3** | 70B | 64 | 4 | 4 | 2 | 1 | TP+PP |
| **LLama-3.1** | 405B | 1024 | 8 | 8 | 2 | 1 | 3D parallelism for scale |
| **GPT-3** | 175B | 128-512 | 4 | 8 | 1 | 1 | Large model config |
| **Mixtral** | 8x7B | 64 | 1 | 4 | 1 | 8 | EP for MoE |
| **Mixtral** | 8x22B | 256 | 4 | 4 | 8 | 8 | Combined TP+EP for large MoE |
| **DeepSeek-V3** | 671B | 1024 | 2 | 16 | 1 | 64 | Large MoE config |

### MoE-Specific Requirements

**Important**: When combining Expert Parallelism (EP) with Tensor Parallelism (TP), **Sequence Parallelism (SP) must be enabled**.

## Performance Optimizations

| Feature                     | Flag                     | Benefit                                |
|-----------------------------|--------------------------|----------------------------------------|
| FlashAttention              | `--attention-backend`    | Faster attention, lower memory usage    |
| FP8 Training                | `--fp8-hybrid`           | Faster training                        |
| Activation Checkpointing    | `--recompute-activations`| Reduced memory usage                   |
| DP Communication Overlap    | `--overlap-grad-reduce`  | Faster distributed training              |
| Distributed Optimizer       | `--use-distributed-optimizer`| Reduced checkpointing time             |

**→ [NVIDIA NeMo Framework Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html#performance-tuning-guide)** - Comprehensive performance optimization guide covering advanced tuning techniques, communication overlaps, memory optimizations, and profiling options.

### FlashAttention

[FlashAttention](https://github.com/Dao-AILab/flash-attention) provides fast, memory-efficient attention.  The default uses cuDNN via Transformer Engine for up to 50% speedups on forward and 84% on backward propagation with FP8 kernels. Also supports `--use-flash-attn`.

### Mixed Precision Training

```bash
--fp16                    # Standard FP16
--bf16                    # BFloat16 (recommended for large models)
--fp8-hybrid              # FP8 training (Hopper, Ada, and Blackwell GPUs)
```

### Activation Checkpointing and Recomputation

```bash
# For limited memory
--recompute-activations

# For extreme memory constraints
--recompute-granularity full \
--recompute-method uniform
```

### Data Parallelism Communication Overlap

```bash
--overlap-grad-reduce
--overlap-param-gather
```

### Distributed Optimizer

```bash
--use-distributed-optimizer
```

## Roadmaps

*   **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - MoE feature development including DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements.
*   **[GPT-OSS Implementation Tracker](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Includes YaRN RoPE scaling, attention sinks, and custom activation functions.

## Community & Support

### Getting Help

*   📖 **[Documentation](https://docs.nvidia.com/Megatron-Core/)**
*   🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)**

### Contributing

We ❤️ contributions!
*   Report bugs.
*   Suggest features.
*   Improve documentation.
*   Submit PRs.

**→ [Contributing Guide](./CONTRIBUTING.md)**

### Citation

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```