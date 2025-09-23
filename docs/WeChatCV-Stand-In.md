<div align="center">
  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Transform your text into stunning videos while preserving identity with Stand-In, a cutting-edge and easy-to-use framework.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)
</div>

[**Explore the Stand-In Repository on GitHub**](https://github.com/WeChatCV/Stand-In)

---

## Overview

**Stand-In** revolutionizes video generation by offering a lightweight, plug-and-play solution for preserving the identity of subjects in your videos. This innovative framework requires training only **1%** of the base video generation model's parameters, achieving state-of-the-art results in both Face Similarity and Naturalness. Stand-In seamlessly integrates into various video generation tasks, including subject-driven, pose-controlled, and stylized video creation, and even face swapping.

## Key Features

*   🚀 **Efficient Training:** Achieves impressive results with only 1% of additional parameter training.
*   🖼️ **High-Fidelity Results:** Maintains excellent identity consistency while preserving video quality.
*   🔌 **Plug-and-Play Integration:** Effortlessly integrates with existing Text-to-Video (T2V) models.
*   🔄 **Extensible Functionality:** Compatible with community models like LoRA and supports diverse video tasks.

---

## What's New

*   **[2025.08.18]** Compatibility with VACE released! Experience pose control and other control methods, such as depth maps, combined with Stand-In.
*   **[2025.08.16]** Experimental face swapping feature updated.
*   **[2025.08.13]** Official Stand-In preprocessor ComfyUI node released: [https://github.com/WeChatCV/Stand-In\_Preprocessor\_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI).
*   **[2025.08.12]** Stand-In v1.0 (153M parameters) released, with open-sourced weights adapted for Wan2.1-14B-T2V and inference code.

---

## Showcase: See Stand-In in Action!

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt                                                                                                                                                                                                                                        | Generated Video                                                                                                                                                                                                                           |
| :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      | "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |                                                                                                                                                                                                                                            Image                                                                                                                                                                                                                                   |
|      Image      | "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind."    |                                                                                                                                                                                                                                            Image                                                                                                                                                                                                                                   |

---

### Non-Human Subjects-Preserving Video Generation

| Reference Image | Prompt                                                                                                                                  | Generated Video                                                                                                                                                                                    |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      | "A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads." |                                                                                                                                                                                                   Image                                                                                                                                                                                                  |

---

### Identity-Preserving Stylized Video Generation

| Reference Image | LoRA          | Generated Video                                                                                                                          |
| :-------------: | :------------ | :---------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      | Ghibli LoRA   |                                                                                                                                    Image                                                                                                                                  |

---

### Video Face Swapping

| Reference Video | Identity                                | Generated Video                                                                                                                                                                                                  |
| :-------------: | :-------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      |                Image                |                                                                                                                                                                                                  Image                                                                                                                                                                                                  |

---

### Pose-Guided Video Generation (With VACE)

| Reference Pose | First Frame | Generated Video                                                                                                                                                                                                   |
| :-------------: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      |    Image     |                                                                                                                                                                                                    Image                                                                                                                                                                                                   |

---

**For more exciting results, please visit our project page:** [https://www.Stand-In.tech](https://www.Stand-In.tech)

---

## Getting Started: Quick Setup Guide

### 1.  Environment Setup

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install required dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference
pip install flash-attn --no-build-isolation
```

### 2. Model Download

Download all necessary model weights using the provided script:

```bash
python download_models.py
```

This script automatically downloads:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (the Stand-In model)

> **Note:** If you possess the `wan2.1-T2V-14B` model locally, you can modify the `download_model.py` script to comment out the corresponding download and place your model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## Usage Instructions

### Standard Inference

Generate identity-preserving videos using the `infer.py` script.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tip:** To avoid altering facial features, simply use "a man" or "a woman" in your prompt. Prompts support both Chinese and English. Ideal for frontal, medium-to-close-up videos.

**Input Image Recommendation:** For best results, use high-resolution, frontal face images. Our built-in preprocessing handles all resolution and file type automatically.

---

### Inference with Community LoRA

Load LoRA models alongside Stand-In using the `infer_with_lora.py` script.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend the following stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping (Experimental)

Experiment with face swapping using the `infer_face_swap.py` script.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Important Notes:** Since Wan2.1 lacks inpainting capabilities, face swapping is experimental.

*   **`denoising_strength`**: Higher values redraw the background, leading to more natural face areas (but potentially less background consistency). Lower values preserve the background more, but with potential overfitting.

*   **`--force_background_consistency`**: (Optional) For consistent backgrounds, experiment with `denoising_strength` to avoid potential contour issues.

---

### Inference with VACE

Use the `infer_with_vace.py` script to incorporate Stand-In with VACE for pose-guided generation.

```bash
python infer_with_vace.py \
    --prompt "A woman raises her hands." \
    --vace_path "checkpoints/VACE/" \
    --ip_image "test/input/first_frame.png" \
    --reference_video "test/input/pose.mp4" \
    --reference_image "test/input/first_frame.png" \
    --output "test/output/woman.mp4" \
    --vace_scale 0.8
```

**Required:** Download VACE weights from the VACE repository or specify the path in `vace_path`.

```bash
python download_models.py --vace
```

**Input:** Preprocessed control videos are required. Both `reference_video` and `reference_image` are optional and can be used together.

**VACE Bias:** VACE has a face bias, affecting identity preservation. Adjust `vace_scale` for a balance between motion and identity. With only  `ip_image` and `reference_video`,  a weight of 0.5 may be sufficient.

**Support:** Integrating Stand-In with VACE is advanced. Contact us via issues if you have questions.

---

## Acknowledgements

Our project benefits from these excellent open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

A special thank you to [Binxin Yang](https://binxinyang.github.io/) for his contributions to the dataset.

---

## Citation

Please cite our paper if you find our work useful:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation},
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

---

## Get in Touch

For questions or suggestions, please reach out via [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues). Your feedback is valuable!