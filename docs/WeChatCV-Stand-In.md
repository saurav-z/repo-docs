<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <p><b>Effortlessly generate videos that preserve the identity of your subjects with Stand-In, the plug-and-play solution.</b></p>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Stand-In Video Generation Example" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

[Link to Original Repository: https://github.com/WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In)

---

## Overview

**Stand-In** is a cutting-edge framework designed for **identity-preserving video generation**, offering a streamlined, plug-and-play solution.  By training only a fraction of the parameters (approximately 1%) of the base video generation model, Stand-In achieves state-of-the-art results in both face similarity and naturalness, surpassing methods that require full-parameter training. It seamlessly integrates into various video generation tasks, including subject-driven, pose-controlled, and stylized video creation, as well as face swapping.

## Key Features

*   ✅ **Efficient Training:** Requires training only 1% of the base model parameters.
*   ✅ **High Fidelity:** Achieves outstanding identity consistency while maintaining video generation quality.
*   ✅ **Plug-and-Play Integration:** Easily integrates into existing Text-to-Video (T2V) models.
*   ✅ **Extensible:** Compatible with community models (e.g., LoRA) and supports a wide range of downstream video tasks.

## What's New

*   **[2025.08.18]** Released VACE compatibility, enabling simultaneous pose control and identity preservation.
*   **[2025.08.16]** Updated experimental face swapping feature.
*   **[2025.08.13]** Official Stand-In preprocessing ComfyUI node released:  [https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI)
*   **[2025.08.12]**  Stand-In v1.0 (153M parameters) released, with open-sourced Wan2.1-14B-T2V-adapted weights and inference code.

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
### Explore More

For a comprehensive view of Stand-In's capabilities, visit: [https://stand-in-video.github.io/](https://www.Stand-In.tech)

## Getting Started

### 1. Environment Setup
```bash
# Clone the project repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference
# Note: Make sure your GPU and CUDA version are compatible with Flash Attention
pip install flash-attn --no-build-isolation
```

### 2. Model Download
Use the provided script to automatically download the required models:
```bash
python download_models.py
```
This script will download:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (Stand-In model)

> **Note:** If you have the `wan2.1-T2V-14B` model locally, modify the `download_model.py` script to comment out the download code and place your model in the `checkpoints/wan2.1-T2V-14B` directory.

## Usage

### Standard Inference

Run `infer.py` for identity-preserving text-to-video generation.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tips:**  To avoid altering facial features, use prompts like "a man" or "a woman."  Prompts support both Chinese and English. Designed for frontal, medium-to-close-up videos.

**Input Image Recommendations:**  Use high-resolution, frontal face images for optimal results. The built-in preprocessing pipeline handles resolution and file extensions.

### Inference with Community LoRA

Load community LoRA models with `infer_with_lora.py`:

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend this stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping

Use `infer_face_swap.py` for video face swapping:

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note:**  Face swapping is experimental due to the lack of inpainting in Wan2.1.

Adjust `denoising_strength` to balance background redrawing and face naturalness.  For complete background consistency, enable `--force_background_consistency` and experiment with `denoising_strength` to mitigate potential contour issues.

### Infer with VACE

Use `infer_with_vace.py` for identity-preserving video generation with VACE:

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

Download VACE weights from the VACE repository or specify their path in the `vace_path` parameter.

```bash
python download_models.py --vace
```

Preprocess the input control video using VACE's preprocessing tool. Adjust `vace_scale` to balance motion and identity preservation.  Reduce the weight to 0.5 when only `ip_image` and `reference_video` are provided.  For any issues, please raise them in the issue tracker.

## Acknowledgements

This project utilizes the following open-source resources:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We thank the authors and contributors of these projects.

We also acknowledge [Binxin Yang](https://binxinyang.github.io/) for the original dataset.

## Citation

If you use Stand-In in your research, please cite our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

## Contact

For questions and suggestions, please use the [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues) . We appreciate your feedback!