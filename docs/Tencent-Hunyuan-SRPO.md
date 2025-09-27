# SRPO: Directly Aligning Diffusion Trajectory with Human Preference

**SRPO introduces a novel approach to fine-tuning diffusion models, directly aligning the full diffusion trajectory with human preferences for superior image generation.** [Check out the original repository](https://github.com/Tencent-Hunyuan/SRPO) for more details.

[![arXiv](https://img.shields.io/badge/ArXiv-red?logo=arxiv)](https://arxiv.org/abs/2509.06942)
[![Hugging Face](https://img.shields.io/badge/Model-blue?logo=huggingface)](https://huggingface.co/tencent/SRPO/)
[![Project Page](https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue)](https://tencent.github.io/srpo-project-page/)

## Key Features

*   ✨ **Direct Alignment:** A new sampling strategy for diffusion fine-tuning improves the restoration of noisy images, leading to stable and efficient optimization, especially in early timesteps.
*   🚀 **Faster Training:** Achieve significant performance improvements for FLUX.1.dev in under 10 minutes using analytical gradients with only a single image rollout; also supports training with a small real image dataset (less than 1500 images).
*   🛡️ **Reward Hacking Resistant:** Enhanced training strategy with direct backpropagation on reward signals (like ReFL and DRaFT) and regularization using negative rewards; produces comparable performance with various rewards while preventing reward hacking issues (e.g., color/oversaturation).
*   🕹️ **Controllable Fine-tuning:** Introduces dynamically controllable text conditions for on-the-fly adjustment of reward preferences towards different styles.

## What's New?

*   **[2025.09.12]**: Complete training code release, plus tips and experiences for model training; a standard workflow in ComfyUI.
*   **[2025.09.08]**: Paper, checkpoint, and inference code released.

## Open-Source Plan

*   [X] Training code (released)
*   [ ] Quantized version for the FLUX community
*   [ ] Support for additional models

## Installation

1.  **Create Conda Environment:**
    ```bash
    conda create -n SRPO python=3.10.16 -y
    conda activate SRPO
    bash ./env_setup.sh
    ```
    💡 *Dependencies are similar to DanceGRPO.*

## Model Downloads

1.  **SRPO Model:**
    *   Download the `diffusion_pytorch_model.safetensors` from [Hugging Face](https://huggingface.co/tencent/SRPO/tree/main).
    ```bash
    mkdir ./srpo
    huggingface-cli login
    huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
    ```
2.  **FLUX.1-dev:**
    *   Download FLUX.1-dev from [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev).
    ```bash
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

## Inference

### Using ComfyUI

SRPO is compatible with [ComfyUI](https://github.com/comfyanonymous/ComfyUI).  Load the [SRPO-workflow](comfyui/SRPO-workflow.json) or use the example image below.

![Example](comfyui/SRPO-workflow.png)

### Quick Start Inference

```python
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

### Multi-GPU Inference
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py
```

## Training

### Prepare Training

1.  **Pretrain Model**: Download FLUX.dev.1 checkpoints to `./data/flux` from [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev).

    ```bash
    mkdir data
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```
2.  **Reward Model (HPS-v2.1 & CLIP):** Download HPS-v2.1 and CLIP checkpoints from [Hugging Face](https://huggingface.co/xswu/HPSv2/tree/main).

    ```bash
    mkdir ./data/hps_ckpt
    huggingface-cli login
    huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
    huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
    ```
3.  **(Optional) Reward Model (PickScore):** Download PickScore checkpoint from [Hugging Face](https://huggingface.co/yuvalkirstain/PickScore_v1) to `./data/ps`.

    ```bash
    mkdir ./data/ps
    huggingface-cli login
    python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1  --local_dir ./data/ps
    python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local_dir ./data/clip
    ```

### Prepare Training Data

```bash
# Place prompts in ./prompts.txt
# Pre-extract text embeddings - improves efficiency
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
cp videos2caption2.json  ./data/rl_embeddings
```

### Training Scripts

*   **HPS-v2.1 Reward Model:**
    ```bash
    bash scripts/finetune/SRPO_training_hpsv2.sh
    ```

*   **(Optional) PickScore Reward Model:**
    ```bash
    bash scripts/finetune/SRPO_training_ps.sh
    ```
    > ⚠️ *PickScore might yield suboptimal results due to a mismatch with current control words.*

### Distributed Training

```bash
#!/bin/bash
echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
node_ip=$(paste -sd, /tmp/pssh.hosts)
pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
```

### Supporting Custom Models

1.  Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` for text embeddings.
2.  Adjust `args.vis_sampling_step` for sigma_schedule.
3.  Enable VAE gradient checkpointing to conserve GPU memory.
4.  Disable the inversion branch for reward hacking tests.
5.  SRPO can be applied to tasks such as OCR and image editing with minimal code adjustments.

### Hyperparameter Recommendations

*   **Batch Size:** Larger batches typically increase quality.
*   **Learning Rate:** 1e-5 to 1e-6.
*   **Train Timestep:** Focus on mid-diffusion stages.
*   **Discount_inv & Discount_denoise:** Fine-tune these parameters for structure preservation and color control.

## Acknowledgements

This work builds upon the following projects:

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