# EasyR1: Supercharge Your Multimodal RL Training with Efficiency and Scalability

**EasyR1 is a powerful, open-source framework designed for efficient and scalable Reinforcement Learning (RL) training of multi-modality models.**  Built upon cutting-edge technology, it enables researchers and developers to train advanced models with ease.  [Check out the original repository](https://github.com/hiyouga/EasyR1) to dive deeper.

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

*Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)*

## Key Features of EasyR1

*   **Wide Model Support:** Includes Llama3, Qwen2/2.5/3, and DeepSeek-R1 language models, as well as Qwen2/2.5-VL vision language models.
*   **Diverse Algorithm Options:** Supports GRPO, DAPO, Reinforce++, ReMax, and RLOO RL algorithms.
*   **Flexible Dataset Compatibility:** Works with any text or vision-text dataset formatted as described [here](#custom-dataset).
*   **Training Enhancements:** Offers padding-free training, checkpoint resuming, and integrated logging with Wandb, SwanLab, Mlflow, and Tensorboard.
*   **Efficient Architecture:** Leverages the [HybirdEngine](https://arxiv.org/abs/2409.19256) and vLLM's SPMD mode for optimal performance.

## Getting Started

### Requirements

*   **Python:** 3.9+
*   **Packages:** transformers>=4.51.0, flash-attn>=2.4.3, vllm>=0.8.3

EasyR1 provides a `Dockerfile` for straightforward environment setup.  It's recommended to use the pre-built Docker image:

```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

For environments without Docker, use Apptainer:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

### Hardware Recommendations

*Estimated hardware requirements for full fine-tuning (AMP and BF16 options available):*

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2*24GB | 4*40GB | 8*40GB | 16*80GB | 32*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1*24GB | 1*40GB | 4*40GB |  8*80GB | 16*80GB |

> [!NOTE]
> Enable bf16 training using `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16`. LoRA support will be added in the next update.

## Tutorial: Run Qwen2.5-VL GRPO on Geometry3K Dataset in 3 Steps

Follow these steps to quickly train a Qwen2.5-VL model using GRPO on the [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) dataset.

![image](assets/qwen2_5_vl_7b_geo.png)

### Installation

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

### GRPO Training

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

### Merge Checkpoint (Hugging Face Format)

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

> [!TIP]
>  Use `export HF_ENDPOINT=https://hf-mirror.com` if you encounter issues connecting to Hugging Face.  For SwanLab logging, use `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh`.

## Custom Dataset

Prepare your datasets using the example formats provided:

*   Text: [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k)
*   Image-text: [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
*   Multi-image-text: [hiyouga/journeybench-multi-image-vqa](https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa)
*   Text-image mixed: [hiyouga/rl-mixed-dataset](https://huggingface.co/datasets/hiyouga/rl-mixed-dataset)

## Understanding the GRPO Algorithm

![image](assets/easyr1_grpo.png)

For more details, see [Hugging Face's blog](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).

## Multi-Node Training with Ray

1.  **Start Head Node:**
    ```bash
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    ```

2.  **Start Worker Nodes:**
    ```bash
    ray start --address=<head_node_ip>:6379
    ```

3.  **Check Resources:**
    ```bash
    ray status
    ```

4.  **Run Training:**  Execute your training script on the Ray head node.

See the **[veRL's official doc](https://verl.readthedocs.io/en/latest/start/multinode.html)** for more details about multi-node training and Ray debugger.

## Performance and Benchmarks

See [baselines.md](assets/baselines.md) for detailed performance benchmarks.

## Awesome Projects Using EasyR1

EasyR1 is actively used in cutting-edge research. Here are some projects leveraging the framework:

*   **MMR1**: Advancing the Frontiers of Multimodal Reasoning. [![[code]](https://img.shields.io/github/stars/LengSicong/MMR1)](https://github.com/LengSicong/MMR1)
*   **Vision-R1**: Incentivizing Reasoning Capability in Multimodal Large Language Models. [![[code]](https://img.shields.io/github/stars/Osilly/Vision-R1)](https://github.com/Osilly/Vision-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06749-blue)](https://arxiv.org/abs/2503.06749)
*   **Seg-Zero**: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement. [![[code]](https://img.shields.io/github/stars/dvlab-research/Seg-Zero)](https://github.com/dvlab-research/Seg-Zero) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06520-blue)](https://arxiv.org/abs/2503.06520)
*   **MetaSpatial**: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse. [![[code]](https://img.shields.io/github/stars/PzySeere/MetaSpatial)](https://github.com/PzySeere/MetaSpatial) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.18470-blue)](https://arxiv.org/abs/2503.18470)
*   **Temporal-R1**: Envolving Temporal Reasoning Capability into LMMs via Temporal Consistent Reward. [![[code]](https://img.shields.io/github/stars/appletea233/Temporal-R1)](https://github.com/appletea233/Temporal-R1)
*   **NoisyRollout**: Reinforcing Visual Reasoning with Data Augmentation. [![[code]](https://img.shields.io/github/stars/John-AI-Lab/NoisyRollout)](https://github.com/John-AI-Lab/NoisyRollout) [![[arxiv]](https://arxiv.org/abs/2504.13055-blue)](https://arxiv.org/pdf/2504.13055)
*   **GUI-R1**: A Generalist R1-Style Vision-Language Action Model For GUI Agents. [![[code]](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)](https://github.com/ritzz-ai/GUI-R1) [![[arxiv]](https://arxiv.org/abs/2504.10458-blue)](https://arxiv.org/abs/2504.10458)
*   **R1-Track**: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/Wangbiao2/R1-Track)](https://github.com/Wangbiao2/R1-Track)
*   **VisionReasoner**: Unified Visual Perception and Reasoning via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/dvlab-research/VisionReasoner)](https://github.com/dvlab-research/VisionReasoner) [![[arxiv]](https://arxiv.org/abs/2505.12081-blue)](https://arxiv.org/abs/2505.12081)
*   **MM-UPT**: Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO. [![[code]](https://img.shields.io/github/stars/waltonfuture/MM-UPT)](https://github.com/waltonfuture/MM-UPT) [![[arxiv]](https://arxiv.org/abs/2505.22453-blue)](https://arxiv.org/pdf/2505.22453)
*   **RL-with-Cold-Start**: Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start. [![[code]](https://img.shields.io/github/stars/waltonfuture/RL-with-Cold-Start)](https://github.com/waltonfuture/RL-with-Cold-Start) [![[arxiv]](https://arxiv.org/pdf/2505.22334-blue)](https://arxiv.org/pdf/2505.22334)
*   **ViGoRL**: Grounded Reinforcement Learning for Visual Reasoning. [![[code]](https://img.shields.io/github/stars/Gabesarch/grounded-rl)](https://github.com/Gabesarch/grounded-rl) [![[arxiv]](https://arxiv.org/abs/2505.22334-blue)](https://arxiv.org/pdf/2505.23678)
*   **Revisual-R1**: Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/CSfufu/Revisual-R1)](https://github.com/CSfufu/Revisual-R1) [![[arxiv]](https://arxiv.org/abs/2506.04207-blue)](https://arxiv.org/abs/2506.04207)
*   **SophiaVL-R1**: Reinforcing MLLMs Reasoning with Thinking Reward. [![[code]](https://img.shields.io/github/stars/kxfan2002/SophiaVL-R1)](https://github.com/kxfan2002/SophiaVL-R1) [![[arxiv]](https://arxiv.org/abs/2505.17018-blue)](https://arxiv.org/abs/2505.17018)
*   **Vision-Matters**: Simple Visual Perturbations Can Boost Multimodal Math Reasoning. [![[code]](https://img.shields.io/github/stars/YutingLi0606/Vision-Matters)](https://github.com/YutingLi0606/Vision-Matters) [![[arxiv]](https://arxiv.org/abs/2506.09736-blue)](https://arxiv.org/abs/2506.09736)
*   **VTool-R1**: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. [![[code]](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)](https://github.com/VTOOL-R1/vtool-r1) [![[arxiv]](https://arxiv.org/abs/2505.19255-blue)](https://arxiv.org/abs/2505.19255)
*   **Long-RL**: Scaling RL to Long Sequences. [![[code]](https://img.shields.io/github/stars/NVlabs/Long-RL)](https://github.com/NVlabs/Long-RL) [![[arxiv]](https://arxiv.org/abs/2507.07966-blue)](https://arxiv.org/abs/2507.07966)

## Future Development

*   **Priorities:**
    *   LoRA Support
*   **Potential Features:**
    *   Ulysses parallelism for VLMs
    *   More VLM architectures

> [!NOTE]
> This project does not include scripts for supervised fine-tuning or inference.  Consider using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for these tasks.

### Known Issues and Troubleshooting

*   **Vision language models and ulysses parallelism are not yet compatible.**
*   **Common Errors and Solutions:**

    *   **ValueError: Image features and image tokens do not match:** Increase `data.max_prompt_length` or reduce `data.max_pixels`.
    *   **RuntimeError: CUDA Error: out of memory:** Reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.
    *   **RuntimeError: 0 active drivers ([]). There should only be one:** Uninstall `deepspeed` from your current environment.

## Get Involved

*   Join the [WeChat group](assets/wechat.jpg) to connect with other users.

## Citation

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

Also, cite the original work:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}