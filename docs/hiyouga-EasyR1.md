# EasyR1: Supercharge Your RL Training with a High-Performance Framework

EasyR1 is a cutting-edge, scalable, and efficient Reinforcement Learning (RL) training framework designed for multi-modality models, offering a streamlined approach to RL development; [view the original repo](https://github.com/hiyouga/EasyR1).

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

**Key Features:**

*   **Supported Models:**
    *   Llama3, Qwen2, Qwen2.5, and Qwen3 language models.
    *   Qwen2 and Qwen2.5-VL vision language models.
    *   DeepSeek-R1 distill models.
*   **Supported Algorithms:** GRPO, DAPO, Reinforce++, ReMax, and RLOO.
*   **Flexible Datasets:** Supports any text or vision-text dataset in a specified format.
*   **Efficient Training:** Features padding-free training, checkpoint resuming, and comprehensive tracking with Wandb, SwanLab, Mlflow, and Tensorboard.
*   **Scalable Architecture:** Leverages the power of **[HybirdEngine](https://arxiv.org/abs/2409.19256)** and **[vLLM](https://github.com/vllm-project/vllm)**'s SPMD mode.

**Used By:** [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)

## Getting Started

### Installation

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

### Run a Quick Example

Train a Qwen2.5-VL model on the Geometry3K dataset in just three steps:

1.  **Installation:** (See above)
2.  **GRPO Training:**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

3.  **Merge Checkpoint:**

    ```bash
    python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
    ```

### Environment Setup

**Recommended:** Use the [pre-built Docker image](https://hub.docker.com/r/hiyouga/verl):

```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

**Alternative:** Apptainer (if Docker is not supported):

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

**Software Requirements:**

*   Python 3.9+
*   transformers>=4.51.0
*   flash-attn>=2.4.3
*   vllm>=0.8.3

### Hardware Requirements

*Estimated GPU Memory:*

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2\*24GB | 4\*40GB | 8\*40GB | 16\*80GB | 32\*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1\*24GB | 1\*40GB | 4\*40GB |  8\*80GB | 16\*80GB |

> \[!NOTE]
> Enable bf16 training using `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16`.

## Custom Dataset

Prepare your dataset in a format compatible with the example datasets:

*   Text dataset: [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k)
*   Image-text dataset: [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
*   Multi-image-text dataset: [hiyouga/journeybench-multi-image-vqa](https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa)
*   Text-image mixed dataset: [hiyouga/rl-mixed-dataset](https://huggingface.co/datasets/hiyouga/rl-mixed-dataset)

## Multi-Node Training

1.  **Start Head Node:**

    ```bash
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    ```

2.  **Start Worker Nodes:**

    ```bash
    ray start --address=<head_node_ip>:6379
    ```

3.  **Check Ray Status:**

    ```bash
    ray status
    ```

4.  **Run Training Script (on Head Node):**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

## Other Baselines

*   CLEVR-70k-Counting
*   GeoQA-8k

## Performance Baselines

See [baselines.md](assets/baselines.md).

## Awesome Projects Using EasyR1

*   **MMR1**: Advancing the Frontiers of Multimodal Reasoning. [[code]](https://github.com/LengSicong/MMR1)
*   **Vision-R1**: Incentivizing Reasoning Capability in Multimodal Large Language Models. [[code]](https://github.com/Osilly/Vision-R1) [[arxiv]](https://arxiv.org/abs/2503.06749)
*   ... (and many more, see the original README for full list with links)

## Frequently Asked Questions (FAQs)

*   **ValueError: Image features and image tokens do not match:**  Increase `data.max_prompt_length` or decrease `data.max_pixels`.
*   **RuntimeError: CUDA Error: out of memory:** Reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.
*   **RuntimeError: 0 active drivers ([]). There should only be one.:**  Uninstall `deepspeed`.

## Citation

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## TODO

*   Support LoRA.
*   Support ulysses parallelism for VLMs.
*   Support more VLM architectures.

> \[!NOTE]
> Supervised fine-tuning and inference scripts are not provided in this project. Consider using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).