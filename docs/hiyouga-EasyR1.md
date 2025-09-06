# EasyR1: Train Cutting-Edge RL Models at Scale

**EasyR1 is a powerful, efficient, and scalable framework for training reinforcement learning (RL) models, offering support for multi-modality and state-of-the-art techniques. See the original repo here: [EasyR1](https://github.com/hiyouga/EasyR1).**

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

### Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)

Built upon the foundation of the [veRL](https://github.com/volcengine/verl) project, EasyR1 provides a high-performance training environment, particularly benefiting from the advancements in [**HybridEngine**](https://arxiv.org/abs/2409.19256) and **vLLM**'s SPMD mode.

## Key Features

*   **Multi-Model Support:** Compatible with a diverse range of models, including Llama3, Qwen2/2.5/3, and DeepSeek-R1. Vision-language models such as Qwen2/2.5-VL are also supported.
*   **Advanced Algorithms:** Implements cutting-edge RL algorithms such as GRPO, DAPO, Reinforce++, ReMax, and RLOO.
*   **Flexible Dataset Compatibility:** Supports various text and vision-text datasets in a customizable format.
*   **Efficiency Enhancements:** Includes features like padding-free training, checkpoint resuming, and integration with logging tools (Wandb, SwanLab, Mlflow, Tensorboard).
*   **Scalable Training:** Enables efficient training across multiple GPUs and nodes.

## Getting Started

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/hiyouga/EasyR1.git
    cd EasyR1
    ```

2.  **Install the Package:**
    ```bash
    pip install -e .
    ```

### Docker and Apptainer

EasyR1 provides Docker images and Apptainer support for environment management.

*   **Docker (Recommended):**
    ```bash
    docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
    docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
    ```

*   **Apptainer:**
    ```bash
    apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
    apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
    ```

### Example: Train Qwen2.5-VL GRPO on Geometry3K Dataset (3 Steps)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/hiyouga/EasyR1.git
    cd EasyR1
    pip install -e .
    ```

2.  **Run GRPO Training:**
    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

3.  **Merge Checkpoint:**
    ```bash
    python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
    ```

### Multi-Node Training

1.  **Start Ray Head Node:**
    ```bash
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    ```
2.  **Start Ray Worker Nodes:**
    ```bash
    ray start --address=<head_node_ip>:6379
    ```
3.  **Run Training Script on Head Node:**
    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

## Dataset Preparation

EasyR1 supports training on custom datasets.  See example datasets for format guidance:

*   Text dataset: https://huggingface.co/datasets/hiyouga/math12k
*   Image-text dataset: https://huggingface.co/datasets/hiyouga/geometry3k
*   Multi-image-text dataset: https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa
*   Text-image mixed dataset: https://huggingface.co/datasets/hiyouga/rl-mixed-dataset

## Additional Resources

*   **GRPO Algorithm:**  Understand GRPO with this [Hugging Face blog](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).
*   **Multi-Node Training:** Explore the [veRL's official documentation](https://verl.readthedocs.io/en/latest/start/multinode.html) for details.
*   **Performance Baselines:** Check [baselines.md](assets/baselines.md) for comparative results.

## Known Bugs and TODO

See the original README for known bugs.

-   Support LoRA (high priority).
-   Support ulysses parallelism for VLMs (middle priority).
-   Support more VLM architectures.

## Awesome Works built with EasyR1

A list of projects is listed in the original readme.

## Citation

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}