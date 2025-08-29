<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Effortless Identity Control for Video Generation
  </h1>

  <h3>Generate videos with consistent identities using a lightweight, plug-and-play approach.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

> **Stand-In** is a groundbreaking framework that simplifies identity preservation in video generation. By training only a tiny fraction of additional parameters (just 1%) on top of a base video generation model, it achieves state-of-the-art results in face similarity and naturalness, surpassing methods that require training the entire model. Stand-In seamlessly integrates with various video generation tasks, including subject-driven generation, pose control, video stylization, and face swapping.  Explore the power of Stand-In on GitHub: [https://github.com/WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In)

---

## Key Features

*   **Effortless Identity Preservation:** Maintain the identity of your subject in generated videos with exceptional consistency.
*   **Lightweight & Efficient:** Achieve top-tier results by training only 1% of the base model's parameters, saving time and resources.
*   **Plug-and-Play Integration:** Easily integrate Stand-In with existing text-to-video (T2V) models for immediate results.
*   **Extensive Compatibility:** Works seamlessly with community models like LoRA and supports various downstream video tasks.
*   **Versatile Applications:** Enables subject-driven generation, pose control, video stylization, and face swapping.

---

## What's New

*   **[2025.08.18]** Released a version compatible with VACE, enabling pose control and integration with other control methods like depth maps, all while preserving identity.
*   **[2025.08.16]** Updated the experimental face swapping feature.
*   **[2025.08.13]** Released an official Stand-In preprocessing ComfyUI node to ensure optimal performance within ComfyUI.

---

## Showcase

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt                                                                                                                                                                                                                                   | Generated Video                                                                                                                                                               |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      | "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |                                                                                                                                                                               |
|      Image      | "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." | ![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab) |

---

### Non-Human Subjects-Preserving Video Generation

| Reference Image | Prompt                                                                                                                                                     | Generated Video                                                                                                                                                               |
| :-------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      | "A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads." | ![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5) |

---

### Identity-Preserving Stylized Video Generation

| Reference Image |     LoRA      | Generated Video                                                                                                                                |
| :-------------: | :-----------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      | Ghibli LoRA | ![Image](https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f)  |

---

### Video Face Swapping

| Reference Video |      Identity      | Generated Video                                                                                                                                |
| :-------------: | :----------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      |       Image        | ![Image](https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50) |

---

### Pose-Guided Video Generation (With VACE)

| Reference Pose |  First Frame   | Generated Video                                                                                                                                |
| :-------------: | :------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|      Image      |     Image      | ![Image](https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44) |

---

For more examples and results, please visit our project page: [https://www.Stand-In.tech](https://www.Stand-In.tech)

---

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install required dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference (check compatibility)
pip install flash-attn --no-build-isolation
```

### 2. Model Download

A script is provided to automatically download necessary model weights to the `checkpoints` directory.

```bash
python download_models.py
```

This script downloads:

*   `wan2.1-T2V-14B` (base T2V model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

> **Note:** If you have the `wan2.1-T2V-14B` model locally, you can modify `download_models.py` to skip its download and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## Usage Examples

### Standard Inference

Use `infer.py` for standard identity-preserving text-to-video generation.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tip:** To maintain facial features, use *"a man"* or *"a woman"* without extra details. Prompts support English and Chinese and are optimized for frontal, medium-to-close-up videos.

**Input Image Recommendation:** Use a high-resolution frontal face image. Our built-in preprocessing handles resolution and file extensions.

### Inference with Community LoRA

Use `infer_with_lora.py` to load community LoRA models with Stand-In.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

Recommended stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping

Use `infer_face_swap.py` for video face swapping.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note:** Face swapping is experimental due to limitations with Wan2.1's inpainting capabilities.
Adjust `denoising_strength` for background and face blending. Higher values redraw the background more, while lower values maintain the background more.

### Inference with VACE

Use `infer_with_vace.py` for identity-preserving video generation with VACE.

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

Download VACE weights from the VACE repository or specify the `vace_path`. Preprocess input control video using VACE's tools. Adjust `vace_scale` to balance motion and identity.

---

## Acknowledgements

We are deeply grateful to the following open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for his work on the dataset.

---

## Citation

If you utilize Stand-In in your research, please cite our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

---

## Contact

For any questions or suggestions, please submit an issue on [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues). We value your feedback!