# EasyR1: The Premier Framework for Efficient and Scalable Reinforcement Learning of Multimodal Models

[EasyR1](https://github.com/hiyouga/EasyR1) is a cutting-edge, open-source framework for training multimodal Reinforcement Learning (RL) models, offering unparalleled efficiency and scalability. Leveraging advanced techniques like HybridEngine and vLLM's SPMD mode, EasyR1 empowers researchers and developers to train large language and vision-language models with ease.

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

*   **Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)**

## Key Features

*   **Model Support:**
    *   Llama3, Qwen2, Qwen2.5, and Qwen3 language models
    *   Qwen2 and Qwen2.5-VL vision-language models
    *   DeepSeek-R1 distill models
*   **Algorithms:**
    *   GRPO, DAPO, Reinforce++, ReMax, and RLOO
*   **Dataset Compatibility:** Supports any text or vision-text dataset in a specific format.
*   **Training Enhancements:** Padding-free training, checkpoint resuming, and comprehensive tracking with Wandb, SwanLab, MLflow, and Tensorboard.
*   **Multi-Node Support:**  Enables training of large models across multiple GPUs.
*   **Community Contributions:**  Extensive list of related projects using EasyR1.

## Getting Started

### Requirements

*   Python 3.9+
*   transformers>=4.51.0
*   flash-attn>=2.4.3
*   vllm>=0.8.3

**Pre-built Docker Image (Recommended):**

```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

**Alternative: Apptainer**

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

### Hardware Requirements (Estimated)

See the original README for the detailed hardware requirements table.  BF16 training is enabled with `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16`.

### Tutorial: Training Qwen2.5-VL GRPO

Follow these steps to train a Qwen2.5-VL model using the GRPO algorithm on the Geometry3K dataset:

1.  **Installation:**

    ```bash
    git clone https://github.com/hiyouga/EasyR1.git
    cd EasyR1
    pip install -e .
    ```

2.  **GRPO Training:**

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

3.  **Merge Checkpoint:**

    ```bash
    python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
    ```

    **Tips:**
    *   Use `export HF_ENDPOINT=https://hf-mirror.com` for Hugging Face connection issues.
    *   Use `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh` for SwanLab logging.

## Custom Datasets

EasyR1 supports a variety of datasets. Refer to example datasets for format guidance.

*   Text dataset: [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k)
*   Image-text dataset: [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
*   Multi-image-text dataset: [hiyouga/journeybench-multi-image-vqa](https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa)
*   Text-image mixed dataset: [hiyouga/rl-mixed-dataset](https://huggingface.co/datasets/hiyouga/rl-mixed-dataset)

## Multi-Node Training

EasyR1 is designed for large-scale training.

1.  Start Ray head node:

    ```bash
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    ```

2.  Start Ray worker nodes, connecting to the head node:

    ```bash
    ray start --address=<head_node_ip>:6379
    ```

3.  Check the Ray resource pool.
4.  Run the training script on the Ray head node.

    ```bash
    bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
    ```

For detailed instructions on multi-node training, refer to the [veRL's official documentation](https://verl.readthedocs.io/en/latest/start/multinode.html).

## Related Projects

EasyR1 has been leveraged by many other projects in the field, including the following:

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

## Known Bugs & TODO

*   Support LoRA (high priority).
*   Support ulysses parallelism for VLMs (middle priority).
*   Support more VLM architectures.
*   Vision language models are not compatible with ulysses parallelism yet.

### FAQs and Troubleshooting

*   **ValueError: Image features and image tokens do not match:** Increase `data.max_prompt_length` or reduce `data.max_pixels`.
*   **RuntimeError: CUDA Error: out of memory:** Reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.
*   **RuntimeError: 0 active drivers ([]). There should only be one.:** Uninstall `deepspeed` from the current python environment.

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