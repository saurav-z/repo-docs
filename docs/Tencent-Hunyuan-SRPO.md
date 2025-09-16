# SRPO: Directly Aligning Diffusion Trajectories with Human Preference

**SRPO revolutionizes diffusion fine-tuning, enabling direct alignment of the full diffusion trajectory to generate images that precisely reflect human preferences.** [Explore the original repository](https://github.com/Tencent-Hunyuan/SRPO)

<div align="center">
  <a href='https://arxiv.org/abs/2509.06942'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://huggingface.co/tencent/SRPO/'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://tencent.github.io/srpo-project-page/'><img src='https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue'></a> &nbsp;
</div>

## Key Features

*   **Direct Alignment:** SRPO introduces a novel sampling strategy for diffusion fine-tuning, improving the stability and efficiency of the optimization process, especially during initial timesteps, to restore highly noisy images.
*   **Faster Training:** Achieves significant performance improvements for FLUX.1.dev in under 10 minutes with direct optimization. Supports training with a small dataset of real images.
*   **Reward Hacking Mitigation:** Improves training strategies, reducing reward hacking issues by directly regularizing the model with negative rewards without the need for KL divergence.
*   **Controllable Fine-tuning:** Integrates dynamically controllable text conditions, enabling on-the-fly adjustment of reward preference towards styles within the scope of the reward model.

## What's New

*   **[2025.09.12]:** Complete training code released, along with tips and community discussion.
*   **[2025.09.12]:** Standard workflow available for ComfyUI.
*   **[2025.09.08]:** Paper, checkpoint, and inference code released.

## Open-Source Plan

*   **[X]** Training code (released)
*   **[ ]** Quantized version for the FLUX community.
*   **[ ]** Support for additional models.

## Installation and Dependencies

1.  **Create Conda Environment:**

    ```bash
    conda create -n SRPO python=3.10.16 -y
    conda activate SRPO
    bash ./env_setup.sh
    ```

    The environment setup is similar to DanceGRPO.

## Model Downloads

1.  **Model Cards**

    | Model           | Hugging Face Download URL                                 |
    | :-------------- | :---------------------------------------------------------- |
    | SRPO            | [diffusion\_pytorch\_model](https://huggingface.co/tencent/SRPO/tree/main) |

2.  **Download the SRPO model:**

    ```bash
    mkdir ./srpo
    huggingface-cli login
    huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
    ```

3.  **Download FLUX.1-dev:**
    ```bash
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

## Inference

### ComfyUI

Use the provided workflow in ComfyUI.  Load the following image or the JSON file [SRPO-workflow](comfyui/SRPO-workflow.json)

![Example](comfyui/SRPO-workflow.png)

### Quick Start

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

Replace the `model_path` in `vis.py` and run:

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py
```

## Training

### Prepare Training Model

1.  **Pretrain Model:** Download FLUX.dev.1 checkpoints from Hugging Face to `./data/flux`:

    ```bash
    mkdir data
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

2.  **Reward Model:** Download the HPS-v2.1 (HPS\_v2.1\_compressed.pt) and CLIP H-14 checkpoints from Hugging Face to `./hps_ckpt`:

    ```bash
    mkdir ./data/hps_ckpt
    huggingface-cli login
    huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
    huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
    ```

3.  **(Optional) Reward Model:** Download the PickScore checkpoint from Hugging Face to `./data/ps`:

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

*   **HPS-v2.1:**

    ```bash
    bash scripts/finetune/SRPO_training_hpsv2.sh
    ```

*   **(Optional) PickScore:**

    ```bash
    bash scripts/finetune/SRPO_training_ps.sh
    ```

    >   ⚠️  Control words are designed for HPS-v2.1, so using PickScore may yield suboptimal results.

*   **Run distributed training:**
    ```bash
    #!/bin/bash
    echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
    node_ip=$(paste -sd, /tmp/pssh.hosts)
    pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
    ```

### Supporting Custom Models

1.  Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py`.
2.  Adjust `args.vis_sampling_step`.
3.  Enable VAE gradient checkpointing to reduce GPU memory.
4.  Disable the inversion branch to check for reward hacking.
5.  Pure Direct-Align works for SRPO-unsupported tasks with minimal code changes.

### Hyperparameter Recommendations

Adjust these settings as a starting point:

1.  **Batch\_size:** Larger sizes improve quality. 32 works well for Flux.dev.1.
2.  **Learning\_rate:** 1e-5 to 1e-6.
3.  **Train\_timestep:** Focus on early-to-middle diffusion stages.
4.  **Discount\_inv & Discount\_denoise:** Current hyperparameters are suitable for many models.

## Acknowledgements

*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)
*   [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

## Citation

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