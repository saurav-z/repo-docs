<div align="center">

# Flash Linear Attention: Accelerating Sequence Modeling with Efficient Implementations

[![hf_model](https://img.shields.io/badge/-Models-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/fla-hub)  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/vDaJTmKNcS)

</div>

**Flash Linear Attention (FLA) offers a comprehensive collection of highly optimized, Triton-based implementations for state-of-the-art linear attention models, enabling faster and more efficient sequence modeling.** This repository provides cutting-edge performance with its platform-agnostic design, fully written in PyTorch and Triton and supporting NVIDIA, AMD, and Intel platforms.  [See the original repo](https://github.com/fla-org/flash-linear-attention).

**Key Features:**

*   **Platform-Agnostic & Optimized:**  Leverages PyTorch and Triton for maximum compatibility and performance across various hardware.
*   **Diverse Model Implementations:** Supports a wide range of linear attention models, including RetNet, GLA, Mamba, and more (see [Models](#models)).
*   **Fused Modules:** Offers pre-built, optimized modules like Rotary Embeddings, LayerNorm variations, fused cross-entropy, and KL divergence for faster training.
*   **Flexible Hybrid Models:** Seamlessly integrates standard attention layers into linear attention models for customized architectures.
*   **Easy Integration:**  Provides clear examples for token mixing, fused modules, generation, and training.
*   **Pre-trained Models:**  Access ready-to-use models via the [fla-hub](https://huggingface.co/fla-hub) on Hugging Face.

**Key Capabilities:**

*   [Installation](#installation) - Comprehensive instructions to get you up and running with FLA
*   [Usage](#usage) - Practical examples for token mixing, fused modules, hybrid models, generation, and training.
*   [Models](#models) - An overview of supported linear attention models.
*   [Training](#training) - Guides you through training with our [🔥 `flame`](https://github.com/fla-org/flame) training framework.
*   [Evaluation](#evaluation) - Instructions on how to evaluate your models.
*   [Benchmarks](#benchmarks) - Performance comparisons with other attention mechanisms.
*   [Citation](#citation) - How to cite our work.

## Models

A detailed list of implemented models is available below, sorted by the year the original paper was published.  Training mode recommendations are provided where applicable (e.g., `chunk`).

| Year | Venue   | Model                | Paper                                                                                                                                         | Code                                                                                            |                                                                                                       |
| :--- | :------ | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------: |
| 2023 |         | RetNet               | [Retentive network: a successor to transformer for large language models](https://arxiv.org/abs/2307.08621)                                   | [official](https://github.com/microsoft/torchscale/tree/main)                                   | [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) |
| 2024 | ICML    | GLA                  | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)                                      | [official](https://github.com/berlino/gated_linear_attention)                                   |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gla.py)          |
| 2024 | ICML    | Based                | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)                            | [official](https://github.com/HazyResearch/based)                                               |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/based.py)         |
| 2024 | ACL     | Rebased              | [Linear Transformers with Learnable Kernel Functions are Better In-Context Models](https://arxiv.org/abs/2402.10644)                          | [official](https://github.com/corl-team/rebased/)                                               |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rebased.py)        |
| 2024 | NeurIPS | DeltaNet             | [Parallelizing Linear Transformers with Delta Rule  over Sequence Length](https://arxiv.org/abs/2406.06484)                                   | [official](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py) |      [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py)       |
| 2022 | ACL     | ABC                  | [ABC: Attention with Bounded-memory Control](https://arxiv.org/abs/2110.02488)                                                                |                                                                                                 |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/abc.py)          |
| 2023 | NeurIPS | HGRN                 | [Hierarchically Gated Recurrent Neural Network for Sequence Modeling](https://openreview.net/forum?id=P1TCHxJwLB)                             | [official](https://github.com/OpenNLPLab/HGRN)                                                  |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn.py)         |
| 2024 | COLM    | HGRN2                | [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904)                                                             | [official](https://github.com/OpenNLPLab/HGRN2)                                                 |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn2.py)         |
| 2024 | COLM    | RWKV6                | [Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence](https://arxiv.org/abs/2404.05892)                                    | [official](https://github.com/RWKV/RWKV-LM)                                                     |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rwkv6.py)         |
| 2024 |         | LightNet             | [You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet](https://arxiv.org/abs/2405.21022)                           | [official](https://github.com/OpenNLPLab/LightNet)                                              |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/lightnet.py)       |
| 2025 | ICLR    | Samba                | [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522)                 | [official](https://github.com/microsoft/Samba)                                                  |          [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/samba)          |
| 2024 | ICML    | Mamba2               | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) | [official](https://github.com/state-spaces/mamba)                                               |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mamba2)          |
| 2024 | NeurIPS | GSA                  | [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146)                                          | [official](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)          |           [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)           |
| 2025 | ICLR    | Gated DeltaNet       | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)                                                    | [official](https://github.com/NVlabs/GatedDeltaNet)                                             |      [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule)      |
| 2025 |         | RWKV7                | [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456)                                                    | [official](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)                                |           [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/rwkv7)            |
| 2025 |         | NSA                  | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)                         |                                                                                                 |            [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/nsa)             |
| 2025 | ICLR    | FoX                  | [Forgetting Transformer: Softmax Attention with a Forget Gate](https://arxiv.org/abs/2503.02130)                                              | [official](https://github.com/zhixuan-lin/forgetting-transformer)                               |      [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/forgetting_attn)       |
| 2025 |         | DeltaProduct         | [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297)                            |                                                                                                 |  [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/layers/gated_deltaproduct.py)  |
| 2025 | ICLR    | Rodimus&ast;         | [Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions](https://arxiv.org/abs/2410.06577)                            | [official](https://github.com/codefuse-ai/rodimus)                                              |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rodimus.py)        |
| 2025 |         | MesaNet              | [MesaNet: Sequence Modeling by Locally Optimal Test-Time Training](https://arxiv.org/abs/2506.05233)                                          |                                                                                                 |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mesa_net.py)       |
| 2025 |         | Comba                | [Comba: Improving Bilinear RNNs with Closed-loop Control](https://arxiv.org/abs/2506.02475)                                                   | [official](https://github.com/AwesomeSeq/Comba-triton)                                          |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/comba.py)         |
| 2025 |         | PaTH                 | [PaTH Attention: Position Encoding via Accumulating Householder Transformations](https://arxiv.org/abs/2505.16381)                            |                                                                                                 |      [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/path_attn.py)       |
| 2025 |         | MoM                  | [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/abs/2502.13685)                                                    | [official](https://github.com/OpenSparseLLMs/MoM)                                               |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mom.py)          |
| 2025 |         | Log-Linear Attention | [Log-Linear Attention](https://arxiv.org/abs/2506.04761)                                                                                      | [official](https://github.com/HanGuo97/log-linear-attention)                                    |      [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/log_linear_attn)       |

## Installation

Ensure you have the following dependencies installed:

*   [PyTorch](https://pytorch.org/) >= 2.5
*   [Triton](https://github.com/openai/triton) >=3.0 (or nightly version, see [FAQs](FAQs.md))
*   [einops](https://einops.rocks/)
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0

Install FLA using pip:

```bash
pip install flash-linear-attention
```

To install from source for the latest features, you must uninstall the existing packages first:

```bash
pip uninstall fla-core flash-linear-attention -y && pip install -U git+https://github.com/fla-org/flash-linear-attention
```

## Usage

### Token Mixing

Integrate linear attention layers into your models using the modules in `fla.layers`:

```python
import torch
from fla.layers import MultiScaleRetention
batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 1024
device, dtype = 'cuda:0', torch.bfloat16
retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
y, *_ = retnet(x)
print(y.shape) # torch.Size([32, 2048, 1024])
```

#### Example GLA Model from 🤗 Transformers

```python
from fla.models import GLAConfig
from transformers import AutoModelForCausalLM
config = GLAConfig()
model = AutoModelForCausalLM.from_config(config)
print(model) # Displays model architecture
```

### Fused Modules

Utilize optimized modules from `fla.modules` for enhanced efficiency:

*   `Rotary Embedding`
*   `RMSNorm`, `LayerNorm`
*   `RMSNormLinear`, `LayerNormLinear`
*   `RMSNorm with Gating`
*   `Fused Cross Entropy`
*   `Linear Cross Entropy`
*   `Linear KL Divergence`

Enable `fuse_linear_cross_entropy` in your model configuration to use the fused linear cross entropy (disabled by default).

### Generation

Generate text with pre-trained models using the 🤗 text generation API:

```python
import fla
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'fla-hub/gla-1.3B-100B'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda()
input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

Pre-trained models are available on [`fla-hub`](https://huggingface.co/fla-hub).

### Hybrid Models

Integrate standard attention layers into `fla` models using the `attn` argument in the model configuration.

Example:

```python
from fla.models import SambaConfig
from transformers import AutoModelForCausalLM
config = SambaConfig(num_hidden_layers=2)
config.attn = {
  'layers': [1],
  'num_heads': 18,
  'num_kv_heads': 18,
  'qkv_bias': False,
  'rope_theta': 10000.,
  'window_size': 2048
}
model = AutoModelForCausalLM.from_config(config)
print(model)
```

## Training

For efficient training of `fla` models, leverage the [🔥 `flame`](https://github.com/fla-org/flame) framework built on `torchtitan`. See [the GLA example](https://github.com/fla-org/flash-linear-attention/blob/main/examples/training.md) for details.

## Evaluation

Evaluate models using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library.  Install the library using the instructions on their repo and then run:

```bash
$ MODEL='fla-hub/gla-1.3B-100B'
$ python -m evals.harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
```

or use the accelerate launcher for multi-GPU evaluation.

## Benchmarks

We benchmarked our Triton-based RetNet implementation against CUDA-based FlashAttention2.  The graph below demonstrates the performance on a single H100 GPU for varying sequence lengths.

```py
#  See the benchmarking script for results.
#  You might have to first install `fla` to enable its import via `pip install -e .`
```

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/c2607015-63af-43d1-90d1-ad5fe1670a03">
</div>

## Citation

Please cite our work if you find this repository helpful:

```bib
@software{yang2024fla,
  title  = {FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism},
  author = {Yang, Songlin and Zhang, Yu},
  url    = {https://github.com/fla-org/flash-linear-attention},
  month  = jan,
  year   = {2024}
}

@inproceedings{yang2024gdn,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
  booktitle = {Proceedings of ICLR},
  year      = {2025}
}

@inproceedings{yang2024deltanet,
  title     = {Parallelizing Linear Transformers with the Delta Rule over Sequence Length},
  author    = {Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and Kim, Yoon},
  booktitle = {Proceedings of NeurIPS},
  year      = {2024}
}

@inproceedings{zhang2024gsa,
  title     = {Gated Slot Attention for Efficient Linear-Time Sequence Modeling},
  author    = {Zhang, Yu and Yang, Songlin and Zhu, Ruijie and Zhang, Yue and Cui, Leyang and Wang, Yiqiao and Wang, Bolun and Shi, Freda and Wang, Bailin and Bi, Wei and Zhou, Peng and Fu, Guohong},
  booktitle = {Proceedings of NeurIPS},
  year      = {2024}
}

@inproceedings{qin2024hgrn2,
  title     = {HGRN2: Gated Linear RNNs with State Expansion},
  author    = {Qin, Zhen and Yang, Songlin and Sun, Weixuan and Shen, Xuyang and Li, Dong and Sun, Weigao and Zhong, Yiran},
  booktitle = {Proceedings of COLM},
  year      = {2024}
}

@inproceedings{yang2024gla,
  title     = {Gated Linear Attention Transformers with Hardware-Efficient Training},
  author    = {Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  booktitle = {Proceedings of ICML},
  year      = {2024}
}
```

## Star History

[![Stargazers repo roster for @fla-org/flash-linear-attention](https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=fla-org&repo=flash-linear-attention)](https://github.com/fla-org/flash-linear-attention/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=fla-org/flash-linear-attention&type=Date)](https://star-history.com/#fla-org/flash-linear-attention&Date)

## Acknowledgments

We thank [Bitdeer](https://www.bitdeer.com/) for providing CI server resources.