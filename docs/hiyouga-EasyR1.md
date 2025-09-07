# EasyR1: Train Powerful RL Models Efficiently and at Scale

**Quickly and easily train cutting-edge reinforcement learning models with EasyR1, a high-performance framework.** ([Original Repository](https://github.com/hiyouga/EasyR1))

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

*Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)*

EasyR1 builds upon the foundation of the [veRL](https://github.com/volcengine/verl) project, offering a robust and scalable solution for training multi-modality reinforcement learning models. Leveraging the innovative **[HybirdEngine](https://arxiv.org/abs/2409.19256)** and the efficiency of **[vLLM](https://github.com/vllm-project/vllm)**'s SPMD mode, EasyR1 delivers impressive performance.

## Key Features

*   **Model Support:**
    *   Llama3, Qwen2, Qwen2.5, and Qwen3 language models
    *   Qwen2 and Qwen2.5-VL vision language models
    *   DeepSeek-R1 distill models
*   **Algorithms:** GRPO, DAPO, Reinforce++, ReMax, RLOO
*   **Dataset Flexibility:** Supports any text or vision-text dataset in a [specified format](#custom-dataset).
*   **Training Optimizations:** Padding-free training, checkpoint resumption, and various tracking options including Wandb, SwanLab, Mlflow, and Tensorboard.

## Getting Started

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/hiyouga/EasyR1.git
    cd EasyR1
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -e .
    ```

### Run the Qwen2.5-VL GRPO Example (3 Steps)

This tutorial demonstrates training a Qwen2.5-VL model using the GRPO algorithm on the Geometry3K dataset.

![image](assets/qwen2_5_vl_7b_geo.png)

1.  **Installation** (as above)
2.  **GRPO Training:**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

3.  **Merge Checkpoint:**

    ```bash
    python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
    ```

    *   **Tip:** Use `export HF_ENDPOINT=https://hf-mirror.com` if you experience issues connecting to Hugging Face. For SwanLab integration, use `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh`.

### Docker & Apptainer

*   **Docker (Recommended):** Use the pre-built Docker image for the easiest setup:

    ```bash
    docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
    docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
    ```

*   **Apptainer (Alternative):**

    ```bash
    apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
    apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
    ```

### Hardware Requirements

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2\*24GB | 4\*40GB | 8\*40GB | 16\*80GB | 32\*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1\*24GB | 1\*40GB | 4\*40GB |  8\*80GB | 16\*80GB |

>   **Note:** Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` for BF16 training. LoRA support will be added in a future update to reduce VRAM usage.

## Custom Dataset

Prepare your dataset using examples:

*   Text dataset: [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k)
*   Image-text dataset: [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
*   Multi-image-text dataset: [hiyouga/journeybench-multi-image-vqa](https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa)
*   Text-image mixed dataset: [hiyouga/rl-mixed-dataset](https://huggingface.co/datasets/hiyouga/rl-mixed-dataset)

## Understanding GRPO

![image](assets/easyr1_grpo.png)

*   Learn more about the GRPO algorithm from [Hugging Face's blog](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).

## Multi-Node Training

1.  **Start Ray Head Node:**

    ```bash
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    ```

2.  **Start Ray Worker Nodes:**

    ```bash
    ray start --address=<head_node_ip>:6379
    ```

3.  **Check Ray Resource Pool:**

    ```bash
    ray status
    ```

4.  **Run Training (on head node):**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

    See **[veRL's official doc](https://verl.readthedocs.io/en/latest/start/multinode.html)** for detailed multi-node configuration and Ray debugger information.

## Baselines

Baselines are provided for the following projects:

*   [CLEVR-70k-Counting](examples/baselines/qwen2_5_vl_3b_clevr.sh)
*   [GeoQA-8k](examples/baselines/qwen2_5_vl_3b_geoqa8k.sh)

## Performance

See [baselines.md](assets/baselines.md) for more information.

## Projects Using EasyR1

*   MMR1
*   Vision-R1
*   Seg-Zero
*   MetaSpatial
*   Temporal-R1
*   NoisyRollout
*   GUI-R1
*   R1-Track
*   VisionReasoner
*   MM-UPT
*   RL-with-Cold-Start
*   ViGoRL
*   Revisual-R1
*   SophiaVL-R1
*   Vision-Matters
*   VTool-R1
*   Long-RL

## Future Developments

*   Support for LoRA (high priority)
*   Support for ulysses parallelism for VLMs (middle priority)
*   Support for more VLM architectures.

## FAQs

*   **`ValueError: Image features and image tokens do not match...`**: Increase `data.max_prompt_length` or reduce `data.max_pixels`.
*   **`RuntimeError: CUDA Error: out of memory...`**: Reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.
*   **`RuntimeError: 0 active drivers ([]). There should only be one.`**: Uninstall `deepspeed`.

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