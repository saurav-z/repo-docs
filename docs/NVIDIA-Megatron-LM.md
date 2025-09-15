# Megatron-LM: Train Powerful Transformer Models at Scale

Megatron-LM is a GPU-optimized library from NVIDIA for training large transformer models, enabling cutting-edge performance and scalability. Explore its [original repository](https://github.com/NVIDIA/Megatron-LM) for the latest updates.

## Key Features

*   **High-Performance Training:** Train models with billions of parameters efficiently using advanced GPU optimization techniques.
*   **Distributed Training:** Leverage data, tensor, pipeline, context, and expert parallelism for massive scaling.
*   **Modular Design:** Use composable building blocks to create custom training frameworks with Megatron Core.
*   **FP8 and Mixed Precision:** Leverage FP8 and other mixed-precision training for faster training and lower memory consumption.
*   **Ecosystem Integration:** Compatible with Hugging Face Accelerate, Colossal-AI, DeepSpeed, and other frameworks.
*   **Complete Solution:** Includes reference implementations, training scripts, and examples.
*   **Comprehensive Support:** Backed by extensive documentation, a vibrant community, and regular updates.

## Quick Start

```bash
# 1. Install Megatron Core with required dependencies
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]

# 2. Clone repository for examples
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
```

**[Complete Installation Guide](#installation)**

## What's New

*   **Megatron Bridge:** Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints. ([Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge))
*   **MoE Roadmap:**  Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements. ([MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729))
*   **GPT-OSS Implementation:** Integration of advanced features, including YaRN RoPE scaling, attention sinks, and custom activation functions.  ([GPT-OSS Implementation](https://github.com/NVIDIA/Megatron-LM/issues/1739))
*   **Megatron MoE Model Zoo:** Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models.  ([Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo))
*   **Megatron Core v0.11.0:** New capabilities for multi-data center LLM training. ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/))

<details>
<summary>Previous News</summary>

-   **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
-   **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
-   **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs. Explore the [Megatron Core intro](#Megatron Core) for more details.

</details>

## Project Structure

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

A complete solution for training state-of-the-art models at scale, ideal for research and experimentation.

*   **Best for:**
    *   Training foundation models with cutting-edge performance.
    *   Research on new architectures and training techniques.
    *   Learning distributed training concepts and best practices.
    *   Quick experimentation with proven model configurations.
*   **What you get:**
    *   Pre-configured training scripts.
    *   End-to-end examples.
    *   Research-focused tools.

### Megatron Core: Composable Library

A modular library with GPU-optimized building blocks for creating custom training pipelines.

*   **Best for:**
    *   Framework developers.
    *   Research requiring custom training loops.
    *   ML engineers needing fault-tolerant pipelines.
*   **What you get:**
    *   Composable transformer building blocks.
    *   Advanced parallelism strategies (TP, PP, DP, EP, CP).
    *   Mixed precision support.
    *   Optimized kernels and memory management.
    *   High-performance dataloaders.
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

*   **FP8 Support:** NVIDIA Hopper, Ada, Blackwell GPUs
*   **Recommended:** NVIDIA Turing architecture or later

#### Software Requirements

*   **CUDA/cuDNN/NCCL:** Latest stable versions
*   **PyTorch:** Latest stable version
*   **Transformer Engine:** Latest stable version
*   **Python:** 3.12 recommended

## Performance Benchmarking

For the latest performance results, see [NVIDIA NeMo Framework Performance Summary](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

Our codebase trains models from 2B to 462B parameters across thousands of GPUs, reaching up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

*   **Vocabulary size:** 131,072 tokens
*   **Sequence length:** 4096 tokens
*   **Model scaling:** Varied hidden size, attention heads, and layers
*   **Communication optimizations:** Fine-grained overlapping with DP, TP, and PP

**Key Results:**

*   **6144 H100 GPUs:** Benchmarked 462B parameter model training.
*   **Superlinear scaling:** MFU increases with model size.
*   **End-to-end measurement:** Includes all operations.
*   **Production ready:** Full training pipeline.
*   *Note: Performance results measured without training to convergence*

### Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

### Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

## Training

### Simple Training Example

```bash
# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

### LLama-3 Training Example

```bash
# 8 GPUs, FP8 precision, mock data
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

*   `--input`: Input file path.
*   `--output-prefix`: Prefix for output files.
*   `--tokenizer-type`: Tokenizer type.
*   `--tokenizer-model`: Tokenizer model file.
*   `--workers`: Number of workers.
*   `--append-eod`: Add end-of-document token.

<!-- **→ [Complete Data Preparation Guide](./docs/data-preparation.md)** - Comprehensive guide covering advanced preprocessing, dataset collection, deduplication, and optimization strategies -->

## Parallelism Strategies

### Data Parallelism (DP)

```bash
# Standard DDP - replicate model on each GPU
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

### Fully Sharded Data Parallel (FSDP)

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

| Model         | Size  | GPUs  | TP | PP | CP | EP | Notes                      |
| ------------- | ----- | ----- | -- | -- | -- | -- | -------------------------- |
| **LLama-3**   | 8B    | 8     | 1  | 1  | 2  | 1  | CP for long seqlen (8K)    |
| **LLama-3**   | 70B   | 64    | 4  | 4  | 2  | 1  | TP+PP                      |
| **LLama-3.1** | 405B  | 1024  | 8  | 8  | 2  | 1  | 3D parallelism for scale |
| **GPT-3**     | 175B  | 128-512 | 4  | 8  | 1  | 1  | Large model config         |
| **Mixtral**   | 8x7B  | 64    | 1  | 4  | 1  | 8  | EP for MoE                 |
| **Mixtral**   | 8x22B | 256   | 4  | 4  | 8  | 8  | Combined TP+EP for large MoE |
| **DeepSeek-V3** | 671B | 1024  | 2  | 16 | 1  | 64 | Large MoE config           |

### MoE-Specific Requirements

**Important**: When combining Expert Parallelism (EP) with Tensor Parallelism (TP), **Sequence Parallelism (SP) must be enabled**.

## Performance Optimizations

| Feature                      | Flag                      | Benefit                                 |
| ---------------------------- | ------------------------- | --------------------------------------- |
| **FlashAttention**           | `--attention-backend`      | Faster attention, lower memory usage    |
| **FP8 Training**             | `--fp8-hybrid`            | Faster training                         |
| **Activation Checkpointing** | `--recompute-activations` | Reduced memory usage                    |
| **Data Parallelism Communication Overlap** | `--overlap-grad-reduce` | Faster distributed training           |
| **Distributed Optimizer**    | `--use-distributed-optimizer` | Reduced checkpointing time            |

**[NVIDIA NeMo Framework Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html#performance-tuning-guide)**

### FlashAttention

FlashAttention provides up to 50% speedups on forward and 84% on backward propagation with FP8 kernels via cuDNN. The `flash-attn` package is also supported via `--use-flash-attn`.

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

*   **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Development of MoE features.
*   **[GPT-OSS Implementation Tracker](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Integration of advanced features for GPT-OSS.

*More roadmap trackers will be added soon.*

## Community & Support

### Getting Help

*   📖 **[Documentation](https://docs.nvidia.com/Megatron-Core/)**
*   🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)**

### Contributing

We welcome contributions!

*   🐛 Report bugs.
*   💡 Suggest features.
*   📝 Improve docs.
*   🔧 Submit PRs.

**[Contributing Guide](./CONTRIBUTING.md)**

## Citation

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```