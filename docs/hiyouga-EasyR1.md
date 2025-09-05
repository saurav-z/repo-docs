# EasyR1: Train Powerful Multi-Modal RL Models Efficiently and at Scale

**Easily build and train cutting-edge Reinforcement Learning (RL) models for multi-modal tasks with EasyR1, a highly efficient and scalable framework.  See the original repo [here](https://github.com/hiyouga/EasyR1).**

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

## Key Features

*   **Broad Model Support:**  Compatible with a wide range of language and vision-language models, including Llama3, Qwen2/2.5/3, and DeepSeek-R1 distill models.
*   **Flexible Algorithm Support:** Implements popular RL algorithms such as GRPO, DAPO, Reinforce++, ReMax, and RLOO.
*   **Custom Dataset Compatibility:** Accepts text and vision-text datasets in a flexible, specified format.
*   **Advanced Training Techniques:** Leverages features like padding-free training and checkpoint resumption for efficiency and convenience.
*   **Comprehensive Tracking:** Integrated with Wandb, SwanLab, Mlflow, and Tensorboard for detailed experiment tracking and monitoring.
*   **Efficient Performance:** Designed with HybridEngine ([arXiv:2409.19256](https://arxiv.org/abs/2409.19256)) and vLLM's SPMD mode for high performance.

## Getting Started

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

### Quickstart Tutorial: Training Qwen2.5-VL with GRPO

Train the Qwen2.5-VL 7B model on the Geometry3K dataset using GRPO in just three steps!

1.  **Install and Setup:** Follow the Installation instructions.
2.  **Run the GRPO Training Script:**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

    ![Example Training Image](assets/qwen2_5_vl_7b_geo.png)

3.  **Merge Checkpoint:**

    ```bash
    python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
    ```

    *   **Tip:**  Resolve Hugging Face connection issues by setting `export HF_ENDPOINT=https://hf-mirror.com`.  For SwanLab integration, use `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh`.

## Requirements

*   **Python:** 3.9+
*   **Required Packages:** transformers>=4.51.0, flash-attn>=2.4.3, vllm>=0.8.3
*   **Docker:** Use the provided Dockerfile for environment setup or the pre-built image (`hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0`).
*   **Apptainer:** If Docker is unavailable, consider Apptainer for containerization.

## Hardware Requirements (Estimated)

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2\*24GB | 4\*40GB | 8\*40GB | 16\*80GB | 32\*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1\*24GB | 1\*40GB | 4\*40GB |  8\*80GB | 16\*80GB |

>   **Note:** Enable bf16 training by setting `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16`.  LoRA support is planned for future updates to reduce VRAM usage.

## Custom Dataset

Prepare your custom dataset by referring to the example datasets and a specific format.

*   **Text Dataset:** [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k)
*   **Image-Text Dataset:** [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
*   **Multi-Image-Text Dataset:** [hiyouga/journeybench-multi-image-vqa](https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa)
*   **Text-Image Mixed Dataset:** [hiyouga/rl-mixed-dataset](https://huggingface.co/datasets/hiyouga/rl-mixed-dataset)

## GRPO Explained

For a deeper understanding of the GRPO algorithm, consult [Hugging Face's TRL documentation](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).

![GRPO Overview](assets/easyr1_grpo.png)

## Multi-Node Training

1.  Start the Ray head node:

    ```bash
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    ```

2.  Start the Ray worker node(s) (connect to the head node):

    ```bash
    ray start --address=<head_node_ip>:6379
    ```

3.  Check the Ray resource pool:

    ```bash
    ray status
    ```

4.  Run the training script on the Ray head node only:

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

    See the [veRL's official doc](https://verl.readthedocs.io/en/latest/start/multinode.html) for further details.

## Other Baselines

Replicated baselines of the [R1-V](https://github.com/deep-agent/R1-V) project:

*   [CLEVR-70k-Counting](examples/baselines/qwen2_5_vl_3b_clevr.sh)
*   [GeoQA-8k](examples/baselines/qwen2_5_vl_3b_geoqa8k.sh)

## Performance Baselines

Refer to [baselines.md](assets/baselines.md) for performance data.

## Real-World Applications - Awesome Work Using EasyR1

EasyR1 is being utilized in cutting-edge research:

*   [MMR1](https://github.com/LengSicong/MMR1): Advancing the Frontiers of Multimodal Reasoning
*   [Vision-R1](https://github.com/Osilly/Vision-R1): Incentivizing Reasoning Capability in Multimodal Large Language Models ([arxiv](https://arxiv.org/abs/2503.06749))
*   [Seg-Zero](https://github.com/dvlab-research/Seg-Zero): Reasoning-Chain Guided Segmentation via Cognitive Reinforcement ([arxiv](https://arxiv.org/abs/2503.06520))
*   [MetaSpatial](https://github.com/PzySeere/MetaSpatial): Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse ([arxiv](https://arxiv.org/abs/2503.18470))
*   [Temporal-R1](https://github.com/appletea233/Temporal-R1): Envolving Temporal Reasoning Capability into LMMs via Temporal Consistent Reward
*   [NoisyRollout](https://github.com/John-AI-Lab/NoisyRollout): Reinforcing Visual Reasoning with Data Augmentation ([arxiv](https://arxiv.org/pdf/2504.13055))
*   [GUI-R1](https://github.com/ritzz-ai/GUI-R1): A Generalist R1-Style Vision-Language Action Model For GUI Agents ([arxiv](https://arxiv.org/abs/2504.10458))
*   [R1-Track](https://github.com/Wangbiao2/R1-Track): Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning
*   [VisionReasoner](https://github.com/dvlab-research/VisionReasoner): Unified Visual Perception and Reasoning via Reinforcement Learning ([arxiv](https://arxiv.org/abs/2505.12081))
*   [MM-UPT](https://github.com/waltonfuture/MM-UPT): Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO ([arxiv](https://arxiv.org/pdf/2505.22453))
*   [RL-with-Cold-Start](https://github.com/waltonfuture/RL-with-Cold-Start): Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start ([arxiv](https://arxiv.org/pdf/2505.22334))
*   [ViGoRL](https://github.com/Gabesarch/grounded-rl): Grounded Reinforcement Learning for Visual Reasoning ([arxiv](https://arxiv.org/abs/2505.23678))
*   [Revisual-R1](https://github.com/CSfufu/Revisual-R1): Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning ([arxiv](https://arxiv.org/abs/2506.04207))
*   [SophiaVL-R1](https://github.com/kxfan2002/SophiaVL-R1): Reinforcing MLLMs Reasoning with Thinking Reward ([arxiv](https://arxiv.org/abs/2505.17018))
*   [Vision-Matters](https://github.com/YutingLi0606/Vision-Matters): Simple Visual Perturbations Can Boost Multimodal Math Reasoning ([arxiv](https://arxiv.org/abs/2506.09736))
*   [VTool-R1](https://github.com/VTOOL-R1/vtool-r1): VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use ([arxiv](https://arxiv.org/abs/2505.19255))
*   [Long-RL](https://github.com/NVlabs/Long-RL): Scaling RL to Long Sequences ([arxiv](https://arxiv.org/abs/2507.07966))

## TODO

*   Support LoRA (high priority).
*   Support ulysses parallelism for VLMs (middle priority).
*   Support more VLM architectures.

>   **Note:** Supervised fine-tuning and inference scripts are not provided in this project. For such needs, consider [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Known Bugs

*   Vision language models are not compatible with ulysses parallelism yet.

## Discussion

Join our [WeChat group](assets/wechat.jpg).

## FAQs

*   **`ValueError: Image features and image tokens do not match: tokens: 8192, features 9800`**: Increase `data.max_prompt_length` or reduce `data.max_pixels`.
*   **`RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62`**: Reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.
*   **`RuntimeError: 0 active drivers ([]). There should only be one.`**: Uninstall `deepspeed` from the current python environment.

## Citation

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

Cite the original work:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```