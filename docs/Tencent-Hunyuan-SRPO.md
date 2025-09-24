<!-- Improved README for SRPO -->

<div align="center">
  <h1 align="center">SRPO: Directly Aligning Diffusion Trajectories with Human Preference</h1>
  <p>
    <b>SRPO revolutionizes diffusion models by directly aligning the entire generation trajectory with human preferences for superior image quality.</b>
  </p>
  <div align="center">
    <a href='https://arxiv.org/abs/2509.06942'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
    <a href='https://huggingface.co/tencent/SRPO/'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
    <a href='https://tencent.github.io/srpo-project-page/'><img src='https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue'></a> &nbsp;
    <a href="https://github.com/Tencent-Hunyuan/SRPO"><img src="https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?logo=github"></a>
  </div>
  <div align="center">
    <br>
    Xiangwei Shen<sup>1,2,3*</sup>, Zhimin Li<sup>1*</sup>, Zhantao Yang<sup>1</sup>, Shiyi Zhang<sup>3</sup>, Yingfang Zhang<sup>1</sup>, Donghao Li<sup>1</sup>,
    <br>
    Chunyu Wang<sup>1✝</sup>, Qinglin Lu<sup>1</sup>, Yansong Tang<sup>3,✉️</sup>
    <br>
    <sup>1</sup>Hunyuan, Tencent  <sup>2</sup>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen 
    <sup>3</sup>Shenzhen International Graduate School, Tsinghua University 
    <sup>*</sup>Equal contribution  <sup>✝</sup>Project lead  <sup>✉️</sup>Corresponding author
  </div>
  <br>
  <img src="assets/head.jpg" alt="SRPO Example" width="800"/>
</div>

## Key Features

*   🚀 **Direct Alignment for Stable Optimization:** SRPO introduces a novel sampling strategy for diffusion fine-tuning, resulting in more stable and efficient training, especially in the early diffusion steps.
*   ⚡️ **Faster Training:** Achieve significant performance improvements in under 10 minutes using a single image rollout and analytical gradients.  Training can be accelerated with as few as 1500 real images.
*   🛡️ **Reward Hacking Resistant:**  Improved training strategies to directly regularize the model using negative rewards, improving perceptual quality and avoiding common issues like color and oversaturation overfitting.
*   💡 **Controllable Fine-tuning:** Enables dynamic text condition control, allowing for on-the-fly adjustment of reward preferences towards desired styles.

## News

*   **2025.09.12:** 🎉 Complete training code released! Includes helpful tips and discussion in issues.
*   **2025.09.12:** 🎉 Standard workflow available for ComfyUI users.
*   **2025.09.08:** 🎉 Paper, checkpoints, and inference code released.

## Open-Source Plan

*   [X] Training code (released)
*   [ ] Quantized version for the FLUX community
*   [ ] Support for additional models

## Installation and Dependencies

```bash
conda create -n SRPO python=3.10.16 -y
conda activate SRPO
bash ./env_setup.sh
```

## Model Downloads

1.  **Model Cards:**

    | Model                                   | Hugging Face Download URL                                                                    |
    | :--------------------------------------- | :------------------------------------------------------------------------------------------ |
    | SRPO                                    | [diffusion\_pytorch\_model](https://huggingface.co/tencent/SRPO/tree/main)                |

2.  **Download Model:**

    ```bash
    mkdir ./srpo
    huggingface-cli login
    huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
    ```

3.  **Download FLUX Cache (or use FLUX.1-dev):**

    ```bash
    mkdir ./data/flux
    huggingface-cli login
    huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
    ```

## Inference

### ComfyUI

Use the provided ComfyUI workflow image, or load the JSON file [SRPO-workflow](comfyui/SRPO-workflow.json).

![ComfyUI Workflow Example](comfyui/SRPO-workflow.png)

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

### Inference Script

Run the inference script, replacing `model_path`:

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

2.  **Reward Model (HPS-v2.1):** Download the HPS-v2.1 and CLIP H-14 checkpoints from [Hugging Face](https://huggingface.co/xswu/HPSv2/tree/main) to `./hps_ckpt`.

    ```bash
    mkdir ./data/hps_ckpt
    huggingface-cli login
    huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
    huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
    ```

3.  **(Optional) Reward Model (PickScore):** Download the PickScore checkpoint from [Hugging Face](https://huggingface.co/yuvalkirstain/PickScore_v1) to `./data/ps`.

    ```bash
    mkdir ./data/ps
    huggingface-cli login
    python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1  --local-dir ./data/ps
    python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ./data/clip
    ```

### Prepare Training Data

```bash
# Write training prompts into ./prompts.txt.  Online RL only needs inference text, not image-text pairs.
via ./prompts.txt
# Pre-extract text embeddings from your custom training dataset—this boosts training efficiency.
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

    >   ⚠️ Current control words are designed for HPS-v2.1, so training with PickScore may yield suboptimal results due to this mismatch.

*   **Distributed Training (using pdsh):**

    ```bash
    #!/bin/bash
    echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
    node_ip=$(paste -sd, /tmp/pssh.hosts)
    pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
    ```

### Custom Model Support

1.  Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` for custom training dataset text embeddings.
2.  Adjust `args.vis_sampling_step` to match the model's inference steps.
3.  Enable VAE gradient checkpointing to reduce memory usage.
4.  If implementing outside of FastVideo, disable the inversion branch.
5.  Pure Direct-Align works for tasks (e.g., OCR, Image Editing) with minimal code changes.

### Hyperparameter Recommendations

*   **Batch\_size:**  Larger sizes (e.g., 32) generally improve quality.
*   **Learning\_rate:**  1e-5 to 1e-6.
*   **Train\_timestep:** Focus on early-to-middle diffusion stages.
*   **Discount\_inv & Discount\_denoise:**  Experiment with these parameters.

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