# SRPO: Directly Aligning Diffusion Trajectories with Human Preference

**SRPO introduces a novel method to directly align the entire diffusion trajectory with human preferences, achieving state-of-the-art results in image generation.** ([Original Repository](https://github.com/Tencent-Hunyuan/SRPO))

[![arXiv](https://img.shields.io/badge/ArXiv-red?logo=arxiv)](https://arxiv.org/abs/2509.06942)
[![Hugging Face Model](https://img.shields.io/badge/Model-blue?logo=huggingface)](https://huggingface.co/tencent/SRPO/)
[![Project Page](https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue)](https://tencent.github.io/srpo-project-page/)

**Authors:** Xiangwei Shen<sup>1,2,3*</sup>, Zhimin Li<sup>1*</sup>, Zhantao Yang<sup>1</sup>, Shiyi Zhang<sup>3</sup>, Yingfang Zhang<sup>1</sup>, Donghao Li<sup>1</sup>, Chunyu Wang<sup>1✝</sup>, Qinglin Lu<sup>1</sup>, Yansong Tang<sup>3,✉️</sup>
<sup>1</sup>Hunyuan, Tencent; <sup>2</sup>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen; <sup>3</sup>Shenzhen International Graduate School, Tsinghua University; <sup>*</sup>Equal contribution; <sup>✝</sup>Project lead; <sup>✉️</sup>Corresponding author

![SRPO Example Image](assets/head.jpg)

## Key Features

*   ✅ **Direct Alignment:** A new sampling strategy optimizes the diffusion process for more stable and computationally efficient training, especially in early timesteps, enabling faster results.
*   ⚡️ **Faster Training:** Achieve significant performance improvements for FLUX.1.dev in under 10 minutes of training.  Training supports using a small dataset of real images (fewer than 1500) to replace online rollouts, significantly accelerating training.
*   🛡️ **Reward Hacking Resistance:**  Improved training strategy avoids reward hacking, such as overfitting to color or oversaturation preferences. Uses negative rewards directly, eliminating the need for KL divergence or a separate reward system.
*   ⚙️ **Controllable Fine-tuning:**  Dynamically controllable text conditions enable on-the-fly adjustments of reward preferences, enhancing style control within the reward model's scope.

## What's New

*   **[2025.09.12]**:  Complete training code, plus training tips and experiences released; issue discussions welcome!
*   **[2025.09.12]**:  Standard workflow available for ComfyUI.
*   **[2025.09.08]**:  Paper, checkpoint, and inference code released.

## Open-Source Plan

*   [X] Training code (released!)
*   [ ] Quantized version for the FLUX community.
*   [ ] Support for other models.

## Installation

```bash
conda create -n SRPO python=3.10.16 -y
conda activate SRPO
bash ./env_setup.sh
```
*(Note: Environment dependencies are similar to DanceGRPO.)*

## Download Models

1.  **Model Cards:**

    | Model           | Hugging Face Download URL                               |
    | --------------- | ------------------------------------------------------ |
    | SRPO            | [diffusion\_pytorch\_model](https://huggingface.co/tencent/SRPO/tree/main) |

2.  **Download SRPO Model:**

    ```bash
    mkdir ./srpo
    huggingface-cli login
    huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
    ```

3.  **Load FLUX Cache (or use pre-trained):**

    ```bash
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

## Inference

### Using ComfyUI

Use the provided workflow in ComfyUI.  Load the following image or the JSON file: [SRPO-workflow](comfyui/SRPO-workflow.json)

![ComfyUI Workflow Example](comfyui/SRPO-workflow.png)

### Quick Start Inference (Python)

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

*For example inference, see `vis.py` and replace the `model_path`.*

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py
```

## Training

### Prepare Training Model

1.  **Pretrain Model:** Download FLUX.dev.1 checkpoints ([Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev)) to `./data/flux`.
    ```bash
    mkdir data
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

2.  **Reward Model (HPS-v2.1):** Download HPS-v2.1 and CLIP H-14 checkpoints ([Hugging Face](https://huggingface.co/xswu/HPSv2/tree/main)) to `./data/hps_ckpt`.
    ```bash
    mkdir ./data/hps_ckpt
    huggingface-cli login
    huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
    huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
    ```

3.  **(Optional) Reward Model (PickScore):** Download PickScore checkpoints ([Hugging Face](https://huggingface.co/yuvalkirstain/PickScore_v1)) to `./data/ps`.  Also, download the CLIP-ViT-H-14 checkpoints ([Hugging Face](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)) to `./data/clip`.
    ```bash
    mkdir ./data/ps
    huggingface-cli login
    python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1  --local-dir ./data/ps
    python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ./data/clip
    ```

### Prepare Training Data

```bash
# Create ./prompts.txt with training prompts (inference text only).
# Pre-extract text embeddings for efficiency.
via ./prompts.txt
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
cp videos2caption2.json  ./data/rl_embeddings
```

### Full-Parameter Training

*   **HPS-v2.1 Reward Model:**
    ```bash
    bash scripts/finetune/SRPO_training_hpsv2.sh
    ```
*   **(Optional) PickScore Reward Model:**
    ```bash
    bash scripts/finetune/SRPO_training_ps.sh
    ```
    > ⚠️ Training with PickScore may yield suboptimal results compared to HPS due to control word mismatches.

*   **Distributed Training (pdsh):**
    ```bash
    #!/bin/bash
    echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
    node_ip=$(paste -sd, /tmp/pssh.hosts)
    pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
    ```

### Custom Model Support

1.  Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` for custom datasets.
2.  Adjust `args.vis_sampling_step` to match the model's inference steps.
3.  Enable VAE gradient checkpointing to reduce memory usage.
4.  Disable the inversion branch to check for reward hacking.
5.  Pure Direct-Align is effective for SRPO-unsupported tasks with minimal modifications.

### Hyperparameter Recommendations

*   **Batch_size:** Larger sizes generally improve quality. (e.g., 32 for Flux.dev.1)
*   **Learning_rate:** 1e-5 to 1e-6 is typically effective.
*   **Train_timestep:** Focus on early-to-middle diffusion stages to avoid structural and color artifacts.
*   **Discount_inv & Discount_denoise:** Fine-tune these for structural preservation and color control.

## Acknowledgements

We gratefully acknowledge the contributions of the following projects:

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