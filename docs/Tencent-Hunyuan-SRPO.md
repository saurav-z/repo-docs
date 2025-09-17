# SRPO: Directly Aligning Diffusion Trajectory with Human Preference

**SRPO (pronounced "Super-Poe") revolutionizes image generation by directly aligning the full diffusion trajectory with fine-grained human preferences, leading to faster training and improved image quality.** Learn more at the [original GitHub repository](https://github.com/Tencent-Hunyuan/SRPO).

<div align="center">
  <a href='https://arxiv.org/abs/2509.06942'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://huggingface.co/tencent/SRPO/'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://tencent.github.io/srpo-project-page/'><img src='https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue'></a> &nbsp;
</div>

## Key Features

*   **Direct Alignment:** A novel sampling strategy for diffusion fine-tuning restores noisy images effectively, optimizing more stably and with reduced computational needs, especially in the initial timesteps.
*   **Faster Training:** Achieves significant performance improvements for FLUX.1.dev in under 10 minutes using analytical gradients and single-image rollouts. Supports training with fewer than 1500 real images by replacing online rollouts.
*   **Reward Hacking Resistance:** Improves training strategy for direct backpropagation on reward signals (e.g., ReFL and DRaFT) and regularizes the model using negative rewards without needing KL divergence or separate reward systems, achieving comparable performance across multiple rewards.
*   **Controllable Fine-tuning:** Introduces dynamically controllable text conditions for on-the-fly adjustment of reward preference toward styles, enabling a new level of control within the reward model's scope.

## What's New?

*   **[2025.9.12]** Training code released! Along with tips and experiences to help you train your models, welcome to discuss and ask questions in the issues! 💬✨
*   **[2025.9.12]** Standard workflow provided for use in ComfyUI.
*   **[2025.9.8]** Paper, checkpoint, and inference code released.

## Getting Started

### Dependencies and Installation

```bash
conda create -n SRPO python=3.10.16 -y
conda activate SRPO
bash ./env_setup.sh
```
💡 *Note: Environment dependencies are similar to DanceGRPO.*

### Download Models

1.  **Model Cards:**

    | Model       | Hugging Face Download URL                                                |
    | :---------- | :------------------------------------------------------------------------ |
    | SRPO        | [diffusion\_pytorch\_model](https://huggingface.co/tencent/SRPO/tree/main) |

2.  **Download the SRPO Model:**
    ```bash
    mkdir ./srpo
    huggingface-cli login
    huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
    ```

3.  **Load FLUX Cache:**
    ```bash
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

### Inference

#### Using ComfyUI

1.  Load the provided ComfyUI workflow image in ComfyUI, or load the JSON file directly: [SRPO-workflow](comfyui/SRPO-workflow.json).
    ![Example](comfyui/SRPO-workflow.png)

#### Quick Start
```bash
from diffusers import FluxPipeline
from safetensors.torch import load_file

prompt='The Death of Ophelia by John Everett Millais, Pre-Raphaelite painting, Ophelia floating in a river surrounded by flowers, detailed natural elements, melancholic and tragic atmosphere'
pipe = FluxPipeline.from_pretrained('./data/flux',
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")
state_dict = load_file("./srpo/diffusion_pytorch_model.safetensors")
pipe.transformer.load_state_dict(state_dict)
image = pipe(
    prompt,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=generator
).images[0]
```

Inference with our cases. Replace `model_path` in `vis.py`.
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py
```

### Training

Follow these steps to train your model.

#### Prepare Training Model

1.  Download FLUX.dev.1 checkpoints to `./data/flux` from [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-dev).
    ```bash
    mkdir data
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

2.  Download Reward Model checkpoints to `./hps_ckpt` from [huggingface](https://huggingface.co/xswu/HPSv2/tree/main).
    ```bash
    mkdir ./data/hps_ckpt
    huggingface-cli login
    huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
    huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
    ```

3.  (Optional) Download PickScore checkpoint to `./data/ps` from [huggingface](https://huggingface.co/yuvalkirstain/PickScore_v1).
    ```bash
    mkdir ./data/ps
    huggingface-cli login
    python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1  --local-dir ./data/ps
    python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ./data/clip
    ```

#### Prepare Training Data

```bash
# Write training prompts into ./prompts.txt. Note: For online RL, no image-text pairs are needed—only inference text.
via ./prompts.txt
# Pre-extract text embeddings from your custom training dataset—this boosts training efficiency.
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
cp videos2caption2.json  ./data/rl_embeddings
```

#### Full-parameter Training

*   HPS-v2.1 for Reward Model:
    ```bash 
    bash scripts/finetune/SRPO_training_hpsv2.sh
    ```
*   (Optional) PickScore for Reward Model:
    ```bash
    bash scripts/finetune/SRPO_training_ps.sh
    ```
    > ⚠️ Current control words are designed for HPS-v2.1, so training with PickScore may yield suboptimal results vs. HPS due to this mismatch.

*   Run distributed training with pdsh.
  ```bash
    #!/bin/bash
    echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
    node_ip=$(paste -sd, /tmp/pssh.hosts)
    pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
  ```

#### How to Support Custom Models

1.  Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py`.
2.  Adjust `args.vis_sampling_step`.
3.  Enable VAE gradient checkpointing.
4.  Disable the inversion branch.
5.  Pure Direct-Align works for SRPO-unsupported tasks with minimal code changes.

#### Hyperparameter Recommendations

*   **Batch\_size**:  32 (Flux.dev.1 reinforcement).
*   **Learning\_rate**:  1e-5 to 1e-6.
*   **Train\_timestep**: Focus on early-to-middle diffusion stages.
*   **Discount\_inv** & **Discount\_denoise**: Current hyperparameters are a good baseline.

## Acknowledgements

*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)
*   [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

## Citation

If you use SRPO in your research, please cite it using the following BibTeX:

```
@misc{shen2025directlyaligningdiffusiontrajectory,
      title={Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference}, 
      author={Xiangwei Shen and Zhimin Li and Zhantao Yang and Shiyi Zhang and Yingfang Zhang and Donghao Li and Chunyu Wang and Qinglin Lu and Yansong Tang},
      year={2025},
      eprint={2509.06942},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.06942}, 
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Tencent-Hunyuan/SRPO&type=Date)](https://www.star-history.com/#Tencent-Hunyuan/SRPO&Date)