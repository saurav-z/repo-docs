# EasyR1: Supercharge Your RL Training with an Efficient and Scalable Framework

**EasyR1** empowers researchers and developers to efficiently train multi-modality Reinforcement Learning (RL) models, offering a robust and scalable solution for various applications. Explore the power of EasyR1 on [GitHub](https://github.com/hiyouga/EasyR1).

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

*Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)*

EasyR1 is a refined fork of the [veRL](https://github.com/volcengine/verl) project, optimized for vision language models.  It leverages the power of **[HybirdEngine](https://arxiv.org/abs/2409.19256)** and **[vLLM](https://github.com/vllm-project/vllm)**'s SPMD mode for high-performance RL training.

## Key Features

*   **Model Support:**
    *   Llama3/Qwen2/Qwen2.5/Qwen3 language models
    *   Qwen2/Qwen2.5-VL vision language models
    *   DeepSeek-R1 distill models
*   **Algorithm Support:**
    *   GRPO
    *   DAPO
    *   Reinforce++
    *   ReMax
    *   RLOO
*   **Dataset Compatibility:**
    *   Flexible support for any text or vision-text dataset in a [specific format](#custom-dataset).
*   **Training Enhancements:**
    *   Padding-free training for improved efficiency.
    *   Checkpoint resumption for flexibility.
    *   Wandb, SwanLab, Mlflow, and Tensorboard integration for experiment tracking.

## Getting Started

### Requirements

*   **Python:** 3.9+
*   **Dependencies:** `transformers>=4.51.0`, `flash-attn>=2.4.3`, `vllm>=0.8.3`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/hiyouga/EasyR1.git
    cd EasyR1
    ```

2.  **Install the package:**

    ```bash
    pip install -e .
    ```

### Docker and Apptainer (Alternative for non-Docker environments)

#### Docker

Use the pre-built docker image for easy environment setup:

```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

#### Apptainer

If Docker is unavailable, use Apptainer:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

### Example: Training Qwen2.5-VL GRPO on Geometry3K (3 Steps)

![image](assets/qwen2_5_vl_7b_geo.png)

1.  **Installation** (already covered above)
2.  **GRPO Training**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```
3.  **Merge Checkpoint**

    ```bash
    python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
    ```

> **Tip:** If you encounter issues connecting to Hugging Face, use `export HF_ENDPOINT=https://hf-mirror.com`.  For SwanLab logging, use `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh`.

## Custom Dataset

Prepare your dataset using the example formats:

*   Text: [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k)
*   Image-text: [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
*   Multi-image-text: [hiyouga/journeybench-multi-image-vqa](https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa)
*   Text-image mixed: [hiyouga/rl-mixed-dataset](https://huggingface.co/datasets/hiyouga/rl-mixed-dataset)

## How GRPO Works in EasyR1

![image](assets/easyr1_grpo.png)

Learn more about the GRPO algorithm from [Hugging Face's blog](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).

## Multi-Node Training

1.  **Start Ray Head Node:**

    ```bash
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    ```

2.  **Start Ray Worker Node:**

    ```bash
    ray start --address=<head_node_ip>:6379
    ```

3.  **Check Resource Pool:**

    ```bash
    ray status
    ```

4.  **Run Training Script (on head node only):**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

See the **[veRL's official doc](https://verl.readthedocs.io/en/latest/start/multinode.html)** for details on multi-node training and debugging.

## Other Baselines

We provide baselines based on the [R1-V](https://github.com/deep-agent/R1-V) project:

*   [CLEVR-70k-Counting](examples/baselines/qwen2_5_vl_3b_clevr.sh)
*   [GeoQA-8k](examples/baselines/qwen2_5_vl_3b_geoqa8k.sh)

## Performance Baselines

See [baselines.md](assets/baselines.md).

## Awesome Work using EasyR1

(List of projects leveraging EasyR1)

*   **MMR1**: Advancing the Frontiers of Multimodal Reasoning. [![[code]](https://img.shields.io/github/stars/LengSicong/MMR1)](https://github.com/LengSicong/MMR1)
*   **Vision-R1**: Incentivizing Reasoning Capability in Multimodal Large Language Models. [![[code]](https://img.shields.io/github/stars/Osilly/Vision-R1)](https://github.com/Osilly/Vision-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06749-blue)](https://arxiv.org/abs/2503.06749)
*   **Seg-Zero**: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement. [![[code]](https://img.shields.io/github/stars/dvlab-research/Seg-Zero)](https://github.com/dvlab-research/Seg-Zero) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06520-blue)](https://arxiv.org/abs/2503.06520)
*   **MetaSpatial**: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse. [![[code]](https://img.shields.io/github/stars/PzySeere/MetaSpatial)](https://github.com/PzySeere/MetaSpatial) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.18470-blue)](https://arxiv.org/abs/2503.18470)
*   **Temporal-R1**: Envolving Temporal Reasoning Capability into LMMs via Temporal Consistent Reward. [![[code]](https://img.shields.io/github/stars/appletea233/Temporal-R1)](https://github.com/appletea233/Temporal-R1)
*   **NoisyRollout**: Reinforcing Visual Reasoning with Data Augmentation. [![[code]](https://img.shields.io/github/stars/John-AI-Lab/NoisyRollout)](https://github.com/John-AI-Lab/NoisyRollout) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.13055-blue)](https://arxiv.org/pdf/2504.13055)
*   **GUI-R1**: A Generalist R1-Style Vision-Language Action Model For GUI Agents. [![[code]](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)](https://github.com/ritzz-ai/GUI-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.10458-blue)](https://arxiv.org/abs/2504.10458)
*   **R1-Track**: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/Wangbiao2/R1-Track)](https://github.com/Wangbiao2/R1-Track)
*   **VisionReasoner**: Unified Visual Perception and Reasoning via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/dvlab-research/VisionReasoner)](https://github.com/dvlab-research/VisionReasoner) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.12081-blue)](https://arxiv.org/abs/2505.12081)
*   **MM-UPT**: Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO. [![[code]](https://img.shields.io/github/stars/waltonfuture/MM-UPT)](https://github.com/waltonfuture/MM-UPT) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22453-blue)](https://arxiv.org/pdf/2505.22453)
*   **RL-with-Cold-Start**: Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start. [![[code]](https://img.shields.io/github/stars/waltonfuture/RL-with-Cold-Start)](https://github.com/waltonfuture/RL-with-Cold-Start) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/pdf/2505.22334)
*   **ViGoRL**: Grounded Reinforcement Learning for Visual Reasoning. [![[code]](https://img.shields.io/github/stars/Gabesarch/grounded-rl)](https://github.com/Gabesarch/grounded-rl) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/pdf/2505.23678)
*   **Revisual-R1**: Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/CSfufu/Revisual-R1)](https://github.com/CSfufu/Revisual-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.04207-blue)](https://arxiv.org/abs/2506.04207)
*   **SophiaVL-R1**: Reinforcing MLLMs Reasoning with Thinking Reward. [![[code]](https://img.shields.io/github/stars/kxfan2002/SophiaVL-R1)](https://github.com/kxfan2002/SophiaVL-R1) [![[arxiv]](https://arxiv.org/badge/arxiv-2505.17018-blue)](https://arxiv.org/abs/2505.17018)
*   **Vision-Matters**: Simple Visual Perturbations Can Boost Multimodal Math Reasoning. [![[code]](https://img.shields.io/github/stars/YutingLi0606/Vision-Matters)](https://github.com/YutingLi0606/Vision-Matters) [![[arxiv]](https://arxiv.org/badge/arxiv-2506.09736-blue)](https://arxiv.org/abs/2506.09736)
*   **VTool-R1**: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. [![[code]](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)](https://github.com/VTOOL-R1/vtool-r1) [![[arxiv]](https://arxiv.org/badge/arxiv-2505.19255-blue)](https://arxiv.org/abs/2505.19255)
*   **Long-RL**: Scaling RL to Long Sequences. [![[code]](https://img.shields.io/github/stars/NVlabs/Long-RL)](https://github.com/NVlabs/Long-RL) [![[arxiv]](https://arxiv.org/badge/arxiv-2507.07966-blue)](https://arxiv.org/abs/2507.07966)

## Future Development

### To-Do

*   Support LoRA (high priority).
*   Support ulysses parallelism for VLMs (middle priority).
*   Support more VLM architectures.

> **Note:** Supervised fine-tuning and inference scripts are not provided in this project. Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for those needs.

### Known Issues

*   Vision language models are not compatible with ulysses parallelism yet.

## Frequently Asked Questions (FAQs)

*   **ValueError: Image features and image tokens do not match: tokens: 8192, features 9800**
    *   Increase `data.max_prompt_length` or decrease `data.max_pixels`.
*   **RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62**
    *   Reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.
*   **RuntimeError: 0 active drivers ([]). There should only be one.**
    *   Uninstall `deepspeed` from the current python environment.

## Citation

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

Consider citing the original work:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```