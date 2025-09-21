<!--  SEO-optimized README for SRPO -->
<div align="center" style="font-family: charter;">
  <h1>SRPO: Fine-tuning Diffusion Models with Direct Alignment for Enhanced Image Generation</h1>
  <p><b>SRPO revolutionizes diffusion model training by directly aligning the full diffusion trajectory with human preferences, leading to superior image generation quality.</b></p>
  <div align="center">
    <a href='https://arxiv.org/abs/2509.06942'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv' alt="ArXiv"></a>  &nbsp;
    <a href='https://huggingface.co/tencent/SRPO/'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface' alt="Hugging Face Model"></a> &nbsp; 
    <a href='https://tencent.github.io/srpo-project-page/'><img src='https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue' alt="Project Page"></a> &nbsp;
  </div>
  <div align="center">
    Xiangwei Shen<sup>1,2,3*</sup>, Zhimin Li<sup>1*</sup>, Zhantao Yang<sup>1</sup>, Shiyi Zhang<sup>3</sup>, Yingfang Zhang<sup>1</sup>, Donghao Li<sup>1</sup>, Chunyu Wang<sup>1✝</sup>, Qinglin Lu<sup>1</sup>, Yansong Tang<sup>3,✉️</sup>
  </div>
  <div align="center">
    <sup>1</sup>Hunyuan, Tencent &nbsp;
    <br>
    <sup>2</sup>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen &nbsp;
    <br>
    <sup>3</sup>Shenzhen International Graduate School, Tsinghua University &nbsp;
    <br>
    <sup>*</sup>Equal contribution &nbsp;
    <sup>✝</sup>Project lead &nbsp;
    <sup>✉️</sup>Corresponding author
  </div>
  <br>
  <img src="assets/head.jpg" alt="SRPO Example Image" width="800">
</div>

## Key Features

*   **Direct Alignment:** SRPO introduces a novel sampling strategy for diffusion fine-tuning, enabling the effective restoration of highly noisy images. This approach results in a more stable and computationally efficient optimization process, particularly during the initial timesteps.
*   **Faster Training:** Achieves significant performance improvements for FLUX.1.dev in under 10 minutes of training by optimizing directly with analytical gradients, using only a single image. The training process can be accelerated further by replacing online rollouts with a small dataset of real images (fewer than 1500 images).
*   **Reward Hacking Mitigation:** SRPO's improved training strategy directly regularizes the model using negative rewards, eliminating the need for KL divergence or a separate reward system. It achieves comparable performance across multiple reward models, resulting in improved perceptual quality without the common issues of reward hacking (e.g., overfitting to colors or oversaturation).
*   **Controllable Fine-tuning:** Incorporates dynamically controllable text conditions, providing the ability to adjust reward preferences on the fly towards specific styles within the reward model's scope.

## News

*   **[2025.09.12]:** 🎉 Complete training code released. Includes tips and experiences to guide your model training. Discuss and ask questions via issues! 💬✨
*   **[2025.09.12]:** 🎉 Provides a standard workflow, ready for use in ComfyUI.
*   **[2025.09.08]:** 🎉 Paper, checkpoint, and inference code released.

## Open-source Plan

*   [X] Training code is under review and will be open-sourced by the end of the week.
*   [ ] Release a quantized version for the FLUX community.
*   [ ] Extend support to other models.

## Installation and Dependencies

```bash
conda create -n SRPO python=3.10.16 -y
conda activate SRPO
bash ./env_setup.sh
```

**Note:** The environment dependencies are largely the same as DanceGRPO.

## Download Models

1.  **Model Cards**

    | Model       | Hugging Face Download URL                                                                 |
    | :---------- | :---------------------------------------------------------------------------------------: |
    | SRPO        | [diffusion_pytorch_model](https://huggingface.co/tencent/SRPO/tree/main)                  |

2.  Download `diffusion_pytorch_model.safetensors` from [Hugging Face](https://huggingface.co/tencent/SRPO)
    ```bash
    mkdir ./srpo
    huggingface-cli login
    huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
    ```

3.  Load your FLUX cache or use the `black-forest-labs/FLUX.1-dev` [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev)
    ```bash
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

## Inference

### Using ComfyUI

SRPO is compatible with [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

*   **Workflow:** Load the [SRPO-workflow](comfyui/SRPO-workflow.json) JSON file directly into ComfyUI, or use the image file (which contains the workflow information).

    <img src="comfyui/SRPO-workflow.png" alt="ComfyUI Workflow Example" width="800">

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

### Inference with Cases

1.  Replace `model_path` in `vis.py`.
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

2.  **Reward Model:** Download the HPS-v2.1 (HPS\_v2.1\_compressed.pt) and CLIP H-14 checkpoints from [Hugging Face](https://huggingface.co/xswu/HPSv2/tree/main) to `./hps_ckpt`.
    ```bash
    mkdir ./data/hps_ckpt
    huggingface-cli login
    huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
    huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
    ```

3.  **(Optional) Reward Model:** Download the PickScore checkpoint from [Hugging Face](https://huggingface.co/yuvalkirstain/PickScore_v1) to `./data/ps`.
    ```bash
    mkdir ./data/ps
    huggingface-cli login
    python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1  --local-dir ./data/ps
    python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ./data/clip
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

*   **HPS-v2.1:** Use HPS-v2.1 as the Reward Model.
    ```bash
    bash scripts/finetune/SRPO_training_hpsv2.sh
    ```
*   **(Optional) PickScore:** Use PickScore as the Reward Model.
    ```bash
    bash scripts/finetune/SRPO_training_ps.sh
    ```
    > ⚠️ Current control words are designed for HPS-v2.1; training with PickScore may yield suboptimal results due to this mismatch.

*   **Distributed Training:** Run distributed training with pdsh.
    ```bash
    #!/bin/bash
    echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
    node_ip=$(paste -sd, /tmp/pssh.hosts)
    pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
    ```

### Supporting Custom Models

1.  Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` to pre-extract text embeddings for your custom training dataset to improve efficiency.
2.  Adjust `args.vis_sampling_step` to modify the sigma schedule. This value should typically match the model's regular inference steps.
3.  Direct-propagation needs a significant amount of GPU memory. Use VAE gradient checkpointing before reward calculation to reduce this.
4.  When implementing outside FastVideo, disable the inversion branch first to check for reward hacking. Its presence likely indicates an implementation error.
5.  Pure Direct-Align works for SRPO-unsupported tasks (e.g., OCR, Image Editing) with minimal code changes.

### Hyperparameter Recommendations

For best results, customize these settings for your model and dataset:

1.  **Batch_size:** Larger sizes often improve quality. A batch size of 32 is recommended for Flux.dev.1 reinforcement under current settings.
2.  **Learning_rate:** A learning rate between 1e-5 and 1e-6 is generally suitable for most models.
3.  **Train_timestep:** Focus on early-to-middle diffusion stages. Extremely early stages (sigmas > 0.99) can cause structural distortions; late stages may encourage reward hacking based on color.
4.  **Discount_inv & Discount_denoise:** Let discount_inv = [a, b], discount_denoise = [c, d]. Preserve structure by setting c slightly > b (avoids early layout corruption). Fix color oversaturation by setting a slightly > d (tempers aggressive tones). The current hyperparameters provide a solid baseline for most in-house models.

## Acknowledgement

We acknowledge the following works and their contributions:

*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)
*   [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

## BibTeX

If SRPO is beneficial for your research and applications, please cite it using the following BibTeX entry:

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

```