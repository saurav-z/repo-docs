# Megatron-LM: Train Large Language Models at Scale 🚀

**Megatron-LM empowers you to train massive transformer models with GPU-optimized performance, enabling cutting-edge AI research and development.**  [Explore the original repository](https://github.com/NVIDIA/Megatron-LM).

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/Megatron-Core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.12.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

**Key Features:**

*   **High-Performance Training:** Optimized for NVIDIA GPUs, achieving up to 47% Model FLOP Utilization (MFU) on H100 clusters.
*   **Scalability:** Train models from 2B to 462B parameters across thousands of GPUs.
*   **Modular Architecture:**  Megatron Core provides composable building blocks for custom training frameworks.
*   **Parallelism Strategies:** Supports Data, Tensor, Pipeline, Context, and Expert Parallelism for efficient distributed training.
*   **Mixed Precision and Optimizations:**  Leverages FP8, FlashAttention, and other techniques for faster and more memory-efficient training.
*   **Ecosystem Integration:** Compatible with libraries like Hugging Face Accelerate, Colossal-AI, and DeepSpeed.
*   **Model Zoo:** Provides pre-configured training scripts for GPT, LLama, DeepSeek, Qwen, and more.

## Quick Start

1.  **Install Megatron Core:**

```bash
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]
```

2.  **Clone the Repository:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
```

**For detailed installation guides:** [Installation](#installation)

## What's New

*   🔄 **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for Hugging Face and Megatron checkpoints.
*   🗺️ **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Roadmap for MoE features, including DeepSeek-V3 and Qwen3.
*   🚀 **[GPT-OSS Implementation](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Integration of advanced features into Megatron Core.
*   **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Optimized configurations for training MoE models like DeepSeek-V3, Mixtral, and Qwen3.
*   **[Megatron Core v0.11.0]** New capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

## Project Structure

```
Megatron-LM/
├── megatron/                    # Megatron Core (kernels, parallelism, building blocks)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

### Megatron-LM: Reference Implementation

The **reference implementation** includes Megatron Core and everything needed for training state-of-the-art foundation models at scale.

**Best for:**

*   Training foundation models with leading-edge performance.
*   Exploring new architectures and training techniques.
*   Learning distributed training.
*   Quick experimentation with proven configurations.

### Megatron Core: Composable Library

The **composable library** provides GPU-optimized building blocks for custom training frameworks.

**Best for:**

*   Framework developers.
*   Research teams needing custom training loops.
*   ML engineers.

## Installation

### 🐳 Docker (Recommended)

Use the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for optimal compatibility.

```bash
# Run container with mounted directories
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

### Source Installation

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
bash docker/common/install.sh --environment {dev,lts}
```

### System Requirements

*   **Hardware:** NVIDIA Hopper, Ada, and Blackwell GPUs (for FP8 support), NVIDIA Turing architecture or later is recommended
*   **Software:** Latest stable versions of CUDA/cuDNN/NCCL, PyTorch, Transformer Engine, and Python 3.12 recommended.

## Performance Benchmarking

Refer to the [NVIDIA NeMo Framework Performance Summary](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html) for the latest results.

**Key Results:**

*   Up to **47% MFU** on H100 clusters.
*   Benchmarked 462B parameter model training on 6144 H100 GPUs.
*   Superlinear scaling with model size.

### Weak Scaling Results

[Weak scaling results](images/weak_scaling.png) demonstrate superlinear scaling, with MFU increasing with model size.

### Strong Scaling Results

[Strong scaling results](images/strong_scaling.png) show strong scaling for GPT-3 with a slight reduction in MFU at larger scales.

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

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

**→ [Complete Data Preparation Guide](./docs/data-preparation.md)**

## Parallelism Strategies

### Data Parallelism (DP)

*   Standard DDP.
*   Fully Sharded Data Parallel (FSDP): `--use-custom-fsdp`, `--use-torch-fsdp2`.  Various sharding strategies.

### Tensor Parallelism (TP)

*   Split model layers across GPUs: `--tensor-model-parallel-size`.
*   Enable Sequence Parallelism: `--sequence-parallel`.

### Pipeline Parallelism (PP)

*   Split model depth across GPUs: `--pipeline-model-parallel-size`.
*   Virtual Pipeline: `--virtual-pipeline-model-parallel-size`.

### Context Parallelism (CP)

*   Split long sequences across GPUs: `--context-parallel-size`.
*   Comm type, and Hierarchical context parallelism.

### Expert Parallelism (EP)

*   For Mixture of Experts (MoE) models: `--expert-model-parallel-size`.

### Combining Strategies

**Parallelism Selection Guide**

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

When combining EP with TP, **Sequence Parallelism (SP) must be enabled**.

## Performance Optimizations

| Feature | Flag | Benefit |
|---------|------|---------|
| FlashAttention | `--attention-backend` | Faster attention and lower memory usage |
| FP8 Training | `--fp8-hybrid` | Faster training |
| Activation Checkpointing | `--recompute-activations` | Reduced memory usage |
| Data Parallelism Communication Overlap | `--overlap-grad-reduce` | Faster distributed training |
| Distributed Optimizer | `--use-distributed-optimizer` | Reduced checkpointing time |

**→ [NVIDIA NeMo Framework Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html#performance-tuning-guide)**

### Mixed Precision Training

```bash
--fp16
--bf16
--fp8-hybrid
```

### Activation Checkpointing and Recomputation

```bash
--recompute-activations
--recompute-granularity full
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

*   **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)**
*   **[GPT-OSS Implementation Tracker](https://github.com/NVIDIA/Megatron-LM/issues/1739)**

## Community & Support

### Getting Help

*   📖 **[Documentation](https://docs.nvidia.com/Megatron-Core/)**
*   🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)**

### Contributing

*   Report bugs.
*   Suggest features.
*   Improve docs.
*   Submit PRs.

**→ [Contributing Guide](./CONTRIBUTING.md)**

## Citation

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}