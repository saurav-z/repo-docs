# SRPO: Aligning Diffusion with Human Preference for Superior Image Generation

**SRPO introduces a novel approach to directly align diffusion models with human preferences, leading to more stable training, faster results, and enhanced image quality.** ([Original Repo](https://github.com/Tencent-Hunyuan/SRPO))

[![ArXiv](https://img.shields.io/badge/ArXiv-red?logo=arxiv)](https://arxiv.org/abs/2509.06942)
[![Model](https://img.shields.io/badge/Model-blue?logo=huggingface)](https://huggingface.co/tencent/SRPO/)
[![Project Page](https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue)](https://tencent.github.io/srpo-project-page/)

Authors: Xiangwei Shen<sup>1,2,3*</sup>, Zhimin Li<sup>1*</sup>, Zhantao Yang<sup>1</sup>, Shiyi Zhang<sup>3</sup>, Yingfang Zhang<sup>1</sup>, Donghao Li<sup>1</sup>, Chunyu Wang<sup>1✝</sup>, Qinglin Lu<sup>1</sup>, Yansong Tang<sup>3,✉️</sup>
<sup>1</sup>Hunyuan, Tencent 
<sup>2</sup>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen 
<sup>3</sup>Shenzhen International Graduate School, Tsinghua University 
<sup>*</sup>Equal contribution 
<sup>✝</sup>Project lead 
<sup>✉️</sup>Corresponding author

![SRPO Example Image](assets/head.jpg)

## Key Features

*   **Direct Alignment:** A new sampling strategy for diffusion fine-tuning that restores highly noisy images efficiently, optimizing for stability and computational efficiency, especially in the early stages.
*   **Faster Training:** Achieve significant performance improvements for FLUX.1.dev in under 10 minutes of training by optimizing with analytical gradients and training on a single image rollout. Train FLUX.1.dev effectively with as few as 1500 real images.
*   **Reward Hacking Mitigation:**  Improved training strategy that directly regularizes the model using negative rewards, bypassing the need for KL divergence or a separate reward system. Achieve comparable performance with multiple reward functions while preventing reward hacking issues like color oversaturation.
*   **Controllable Fine-tuning:**  Incorporate dynamically controllable text conditions in online RL, allowing on-the-fly adjustment of reward preferences.

## Updates & News

*   **[2025.09.12]**: Complete training code released, along with helpful training tips and community support via issues.
*   **[2025.09.12]**: Standard workflow available for ComfyUI.
*   **[2025.09.08]**: Paper, checkpoint, and inference code released.

## Open-source Plan

*   \[X] Training code open-sourced.
*   \[ ] Quantized version release.
*   \[ ] Support for other models.

## Installation

```bash
conda create -n SRPO python=3.10.16 -y
conda activate SRPO
bash ./env_setup.sh
```

## Download Models

1.  **Model Cards**

    | Model         | Hugging Face Download URL                                       |
    | :------------ | :-------------------------------------------------------------- |
    | SRPO          | [diffusion\_pytorch\_model](https://huggingface.co/tencent/SRPO/tree/main) |

2.  **Download SRPO Model**
    ```bash
    mkdir ./srpo
    huggingface-cli login
    huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
    ```
3.  **Download FLUX.1-dev** (or use your FLUX cache)
    ```bash
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

## Inference

### Using ComfyUI

Use the provided workflow JSON file (SRPO-workflow.json) or load the example image in ComfyUI.

![ComfyUI Workflow Example](comfyui/SRPO-workflow.png)

### Quick Start

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

Inference cases are available in `vis.py`. Replace `model_path` in the script.

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py
```

## Training

### Prepare Training Model

1.  **Pretrain Model:** Download FLUX.dev.1 checkpoints from [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev) to `./data/flux`.
    ```bash
    mkdir data
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```
2.  **Reward Model:** Download HPS-v2.1 and CLIP checkpoints from [Hugging Face](https://huggingface.co/xswu/HPSv2/tree/main) to `./hps_ckpt`.
    ```bash
    mkdir ./data/hps_ckpt
    huggingface-cli login
    huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
    huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
    ```
3.  **(Optional) Reward Model:** Download PickScore checkpoint from [Hugging Face](https://huggingface.co/yuvalkirstain/PickScore_v1) to `./data/ps`.
    ```bash
    mkdir ./data/ps
    huggingface-cli login
    python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1  --local_dir ./data/ps
    python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local_dir ./data/clip
    ```

### Prepare Training Data

```bash
# Write training prompts into ./prompts.txt. Note: For online RL, no image-text pairs are needed—only inference text.
via ./prompts.txt
# Pre-extract text embeddings from your custom training dataset—this boosts training efficiency.
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
cp videos2caption2.json  ./data/rl_embeddings
```

### Full-parameter Training

*   **HPS-v2.1 (Reward Model):**
    ```bash
    bash scripts/finetune/SRPO_training_hpsv2.sh
    ```
*   **(Optional) PickScore (Reward Model):**
    ```bash
    bash scripts/finetune/SRPO_training_ps.sh
    ```
    > ⚠️  Control words are designed for HPS-v2.1; training with PickScore may yield suboptimal results.

*   **Distributed Training:**  Use `pdsh` to run distributed training.
    ```bash
    #!/bin/bash
    echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
    node_ip=$(paste -sd, /tmp/pssh.hosts)
    pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
    ```

### How to Support Custom Models

1.  Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` to pre-extract text embeddings.
2.  Adjust `args.vis_sampling_step` to match the model's inference steps.
3.  Enable VAE gradient checkpointing to reduce GPU memory.
4.  Disable the inversion branch to check for reward hacking.
5.  Pure Direct-Align works for SRPO-unsupported tasks with minimal code changes.

### Hyperparameter Recommendations

Use these as a starting point, adjusting for your model/dataset:

1.  **Batch\_size:** Larger sizes generally improve quality.  32 works well for Flux.dev.1.
2.  **Learning\_rate:** 1e-5 to 1e-6 is typical.
3.  **Train\_timestep:** Focus on early-to-middle diffusion stages.
4.  **Discount\_inv & Discount\_denoise:** Adjust to preserve structure and mitigate color issues.  Current settings are a good baseline.

## Acknowledgements

We built upon the following works:

*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)
*   [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

## BibTeX

```bibtex
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