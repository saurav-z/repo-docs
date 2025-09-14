<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Generate videos that maintain a consistent identity with minimal training, unlocking new creative possibilities.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

> Stand-In is a groundbreaking framework for identity-preserving video generation that is lightweight and plug-and-play.  Achieving state-of-the-art results in face similarity and naturalness with only **1%** additional parameters compared to the base video generation model, Stand-In offers seamless integration into various tasks, including subject-driven and pose-controlled video generation, video stylization, and face swapping.  Learn more at the original repository: [https://github.com/WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In)

---

## Key Features

*   **Efficiency:** Train with only 1% additional parameters, minimizing computational cost.
*   **High Fidelity:** Preserve identity while maintaining high video generation quality.
*   **Plug-and-Play:** Easy integration with existing Text-to-Video (T2V) models.
*   **Extensible:** Works with LoRA and supports diverse video tasks.
*   **Versatile Application**: Supports identity-preserving text-to-video, non-human subjects-preserving video, identity-preserving stylized video, video face swapping, and pose-guided video generation.

---

## What's New

*   **[2025.08.18]** Released VACE compatibility for advanced pose and depth map control.
*   **[2025.08.16]** Face swapping feature updated.
*   **[2025.08.13]** Official Stand-In Preprocessing ComfyUI node released to ensure optimal integration with ComfyUI.
*   **[2025.08.12]** Stand-In v1.0 (153M parameters) released with Wan2.1-14B-T2V-adapted weights and inference code open-sourced.

---

## Showcase

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)| "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3) |
|![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)| "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." |![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab)|

---
### Non-Human Subjects-Preserving Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" />|"A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads."|![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5) |

---

### Identity-Preserving Stylized Video Generation

| Reference Image | LoRA | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1)|Ghibli LoRA|![Image](https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f)|
---

### Video Face Swapping

| Reference Video | Identity | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701)|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" />|![Image](https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50)|

---

### Pose-Guided Video Generation (With VACE)

| Reference Pose | First Frame | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94)|<img width="719" height="415" alt="Image" src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" />|![Image](https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44)|

---

### Explore More Examples

For more stunning results and detailed examples, visit our dedicated project page: [https://www.Stand-In.tech](https://www.Stand-In.tech)

---

## Getting Started

### 1.  Environment Setup

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install the necessary dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for faster inference (check compatibility)
pip install flash-attn --no-build-isolation
```

### 2. Model Download

Download the required models automatically with the following script:

```bash
python download_models.py
```

This script downloads:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

>   **Note:** If you already have the `wan2.1-T2V-14B` model, modify the `download_model.py` script to avoid redownloading it. Place the model in `checkpoints/wan2.1-T2V-14B`.

---

## Usage

### Standard Inference

Generate identity-preserving videos using the `infer.py` script.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompt Tip:**  To avoid altering facial features, use prompts like *"a man"* or *"a woman"* without specifying appearance.  Frontal, medium-to-close-up videos are best.

**Input Image:** High-resolution, frontal face images are recommended. The built-in preprocessing pipeline will handle different resolutions and formats.

---

### Inference with Community LoRA

Load and combine LoRA models with Stand-In using the `infer_with_lora.py` script.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

Recommended stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping

Use the `infer_face_swap.py` script for experimental video face swapping.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note:**  Face swapping is experimental due to the lack of inpainting in Wan2.1. Experiment with `--denoising_strength` to balance background and face consistency.

---

### Infer with VACE (Pose-Guided Generation)

Combine Stand-In with VACE for pose-guided video generation using the `infer_with_vace.py` script.

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

You must download the VACE weights from the [VACE repository](<https://github.com/your-vace-repo>) or provide the `vace_path` parameter.

```bash
python download_models.py --vace
```

Preprocess the input control video using VACE's tool. Adjust `vace_scale` for a balance between motion and identity preservation. When only `ip_image` and `reference_video` are used, reducing the weight to 0.5 is effective.

---

## Acknowledgements

This project leverages these excellent open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for contributions to dataset creation.

---

## Citation

If you find our work useful, please cite our paper:

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

For questions or suggestions, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues).