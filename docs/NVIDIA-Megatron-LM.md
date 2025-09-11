<div align="center">

# Megatron-LM: Train Large Language Models at Scale with GPU-Optimized Performance 

**Supercharge your large language model training with Megatron-LM, a GPU-optimized library from NVIDIA that delivers cutting-edge performance and scalability.**  [Explore the original repo](https://github.com/NVIDIA/Megatron-LM) for the full details.

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/Megatron-Core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.12.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

</div>

## Key Features

*   **GPU-Optimized Performance:** Leverage NVIDIA's optimized kernels and techniques for blazing-fast training on GPUs.
*   **Scalable Training:** Train models with billions of parameters across multiple GPUs and even multiple data centers.
*   **Modular and Composable:** Utilize Megatron Core, a building-block library for creating custom training workflows.
*   **Advanced Parallelism:** Utilize advanced parallelism strategies (DP, TP, PP, CP, EP) to maximize GPU utilization.
*   **Mixed Precision Support:** Accelerate training with FP16, BF16, and FP8 precision, and leverage Transformer Engine.
*   **Mixture of Experts (MoE) Support:** Optimize MoE models.
*   **Rich Ecosystem:** Integrate with leading libraries like Hugging Face Accelerate, TensorRT-LLM, and DeepSpeed.

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

##  What's New

*   🔄 **Megatron Bridge:** Bidirectional converter for Hugging Face and Megatron checkpoints.
*   🗺️ **MoE Roadmap (Q3-Q4 2025):** Advanced MoE features, including DeepSeek-V3, Qwen3, FP8 optimizations, and Blackwell enhancements.
*   🚀 **GPT-OSS Implementation:** Integration of advanced features like YaRN RoPE scaling.
*   [Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo): Optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models.
*   [Megatron Core v0.11.0](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/): New capabilities for multi-data center LLM training.

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
- [Latest News](#latest-news)
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

Megatron provides a comprehensive solution for training large language models. It consists of a reference implementation, Megatron-LM, and a modular, composable library, Megatron Core.

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

**Reference implementation** that includes Megatron Core plus everything needed to train models.

**Best for:**

*   **Training state-of-the-art foundation models** at scale with cutting-edge performance on latest NVIDIA hardware
*   **Research teams** exploring new architectures and training techniques
*   **Learning distributed training** concepts and best practices
*   **Quick experimentation** with proven model configurations

**What you get:**

*   Pre-configured training scripts for GPT, LLama, DeepSeek, Qwen, and more.
*   End-to-end examples from data prep to evaluation
*   Research-focused tools and utilities

### Megatron Core: Production Library

**Composable library** with GPU-optimized building blocks for custom training frameworks.

**Best for:**

*   **Framework developers** building on top of modular and optimized components
*   **Research teams** needing custom training loops, optimizers, or data pipelines
*   **ML engineers** requiring fault-tolerant training pipelines

**What you get:**

*   Composable transformer building blocks (attention, MLP, etc.)
*   Advanced parallelism strategies (TP, PP, DP, EP, CP)
*   Pipeline schedules and distributed optimizers
*   Mixed precision support (FP16, BF16, FP8)
*   GPU-optimized kernels and memory management
*   High-performance dataloaders and dataset utilities
*   Model architectures (LLaMA, Qwen, GPT, Mixtral, Mamba, etc.)

## Ecosystem Libraries

**Libraries used by Megatron Core:**

*   **[Megatron Energon](https://github.com/NVIDIA/Megatron-Energon)** 📣 **NEW!** - Multi-modal data loader (text, images, video, audio) with distributed loading and dataset blending
*   **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)** - Optimized kernels and FP8 mixed precision support
*   **[Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext)** - Fault tolerant training with failure detection and recovery

**Libraries using Megatron Core:**

*   **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Training library with bidirectional Hugging Face ↔ Megatron checkpoint conversion, flexible training loops, and production-ready recipes
*   **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)** - Scalable toolkit for efficient reinforcement learning with RLHF, DPO, and other post-training methods
*   **[NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)** - Enterprise framework with cloud-native support and end-to-end examples
*   **[TensorRT Model Optimizer (ModelOpt)](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** - Model optimization toolkit for quantization, pruning, and distillation

**Compatible with:** [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## Installation

### 🐳 Docker (Recommended)

Use the previous releases of [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for optimal compatibility. These containers have pre-installed dependencies and optimized configurations.

```bash
# Run container with mounted directories
docker run --runtime --nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### Pip Installation

Install `megatron-core` using pip.

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

For development or latest features:

Install `uv` and build dependencies from source.

```bash
export UV_VERSION=0.7.2
export PATH="$HOME/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
export UV_PROJECT_ENVIRONMENT=./venv
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export UV_LINK_MODE=copy

# Clone and install
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Optional: checkout specific release
git checkout core_r0.13.0

bash docker/common/install.sh --environment {dev,lts}
```

### System Requirements

#### Hardware Requirements

*   **FP8 Support**: NVIDIA Hopper, Ada, Blackwell GPUs
*   **Recommended**: NVIDIA Turing architecture or later

#### Software Requirements

*   **CUDA/cuDNN/NCCL**: Latest stable versions
*   **PyTorch**: Latest stable version
*   **Transformer Engine**: Latest stable version
*   **Python**: 3.12 recommended

## Performance Benchmarking

For the latest results, refer to [NVIDIA NeMo Framework Performance Summary](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

Megatron-LM can train models from 2B to 462B parameters with up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

*   **Vocabulary size**: 131,072 tokens
*   **Sequence length**: 4096 tokens
*   **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
*   **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

*   **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
*   **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
*   **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
*   **Production ready**: Full training pipeline with checkpointing and fault tolerance
*   *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Weak scaled results show superlinear scaling.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

Strong scaling of the standard GPT-3 model.

![Strong scaling](images/strong_scaling.png)

## Training

### Getting Started

```bash
# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

```bash
# LLama-3 Training Example (8 GPUs, FP8 precision, mock data)
./examples/llama/train_llama3_8b_fp8.sh
```

## Data Preparation

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

*   `--input`: Path to input JSON/JSONL file
*   `--output-prefix`: Prefix for output binary files (.bin and .idx)
*   `--tokenizer-type`: Tokenizer type (`HuggingFaceTokenizer`, `GPT2BPETokenizer`, etc.)
*   `--tokenizer-model`: Path to tokenizer model file
*   `--workers`: Number of parallel workers for processing
*   `--append-eod`: Add end-of-document token

## Parallelism Strategies

### Data Parallelism (DP)

```bash
# Standard DDP - replicate model on each GPU
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

```bash
# Megatron's optimized FSDP (~15% faster than PyTorch FSDP2)
--use-custom-fsdp

# PyTorch FSDP2
--use-torch-fsdp2

# Sharding strategies
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

## Combining Parallelism Strategies

### Parallelism Selection Guide

Based on [NVIDIA NeMo production configurations](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance/recommended_model_configs):

| Model        | Size | GPUs | TP | PP | CP | EP | Notes                    |
| :----------- | :--- | :--- | :- | :- | :- | :- | :----------------------- |
| **LLama-3**  | 8B   | 8    | 1  | 1  | 2  | 1  | CP for long seqlen (8K) |
| **LLama-3**  | 70B  | 64   | 4  | 4  | 2  | 1  | TP+PP                  |
| **LLama-3.1**| 405B | 1024 | 8  | 8  | 2  | 1  | 3D parallelism for scale |
| **GPT-3**    | 175B | 128-512 | 4  | 8  | 1  | 1  | Large model config      |
| **Mixtral**  | 8x7B | 64   | 1  | 4  | 1  | 8  | EP for MoE               |
| **Mixtral**  | 8x22B | 256  | 4  | 4  | 8  | 8  | Combined TP+EP for MoE |
| **DeepSeek-V3**| 671B | 1024 | 2  | 16 | 1  | 64 | Large MoE config      |

### MoE-Specific Requirements

**Important**: When combining Expert Parallelism (EP) with Tensor Parallelism (TP), **Sequence Parallelism (SP) must be enabled**.

## Performance Optimizations

| Feature                     | Flag                       | Benefit                               |
| :-------------------------- | :------------------------- | :------------------------------------ |
| **FlashAttention**          | `--attention-backend`      | Faster attention, lower memory usage   |
| **FP8 Training**            | `--fp8-hybrid`             | Faster training                       |
| **Activation Checkpointing** | `--recompute-activations`  | Reduced memory usage                  |
| **Data Parallelism Communication Overlap** | `--overlap-grad-reduce` | Faster distributed training         |
| **Distributed Optimizer**   | `--use-distributed-optimizer` | Reduced checkpointing time            |

**→ [NVIDIA NeMo Framework Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html#performance-tuning-guide)** - Comprehensive performance optimization guide.

### FlashAttention

We recommend the default usage, which uses cuDNN for attention via Transformer Engine, and provides up to 50% speedups on forward and 84% on backward propagation with FP8 kernels.

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

Stay up-to-date with our development roadmaps and planned features:

*   **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - MoE feature development
*   **[GPT-OSS Implementation Tracker](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features

*More roadmap trackers will be added soon.*

## Community & Support

### Getting Help

*   📖 **[Documentation](https://docs.nvidia.com/Megatron-Core/)** - Official documentation
*   🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

### Contributing

We ❤️ contributions!

*   🐛 **Report bugs**
*   💡 **Suggest features**
*   📝 **Improve docs**
*   🔧 **Submit PRs**

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