<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Effortless Identity Control for Video Generation
  </h1>

  <h3>Generate stunning videos while preserving the identity of your subject with Stand-In, a lightweight and plug-and-play solution.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

> **Stand-In** offers a simple and efficient way to maintain subject identity in your video creations.  By using just **1%** extra parameters, Stand-In achieves state-of-the-art performance, exceeding methods requiring full parameter training.  It's easily integrated with other tasks like subject-driven and pose-controlled generation, video stylization, and face swapping.  Find the original repository at [https://github.com/WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In).

---

## Key Features of Stand-In

*   🎯 **Lightweight & Efficient:** Trains with only 1% of additional parameters.
*   🖼️ **Exceptional Fidelity:** Maintains strong identity consistency without sacrificing video quality.
*   🔌 **Plug-and-Play Integration:** Seamlessly integrates with existing Text-to-Video (T2V) models.
*   ✨ **Extensive Compatibility:** Supports community models (LoRA) and various video generation tasks, including:
    *   Text-to-Video Generation
    *   Non-Human Subject-Preserving Video Generation
    *   Stylized Video Generation
    *   Video Face Swapping
    *   Pose-Guided Video Generation (with VACE)

---

## Showcase: See Stand-In in Action!

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
For more examples and results, visit the project page at [https://www.Stand-In.tech](https://www.stand-in.tech)

---

## News and Updates

*   **[2025.08.18]** Compatible with VACE release for pose control and other control methods, allowing for simultaneous identity maintenance.
*   **[2025.08.16]** Experimental face swapping feature updated.
*   **[2025.08.13]** Official Stand-In preprocessing ComfyUI node released to address compatibility issues with custom nodes.  Use our official preprocessing node for best results when using Stand-In within ComfyUI.
*   **[2025.08.12]** Stand-In v1.0 (153M parameters) released, including open-sourced Wan2.1-14B-T2V-adapted weights and inference code.

---

## Getting Started: Quick Setup Guide

### 1. Set up your environment:

```bash
# Clone the project
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install required packages
pip install -r requirements.txt

# (Optional) Speed up inference with Flash Attention
pip install flash-attn --no-build-isolation
```

### 2. Download the necessary models:

```bash
python download_models.py
```

This script downloads:

*   `wan2.1-T2V-14B` (base T2V model)
*   `antelopev2` (face recognition)
*   `Stand-In` (the Stand-In model)

> **Note:** Customize the `download_model.py` script if you have the `wan2.1-T2V-14B` model already.

---

## Usage Instructions

### 1. Standard Inference:

Use `infer.py` to generate identity-preserving videos.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tip:** To avoid altering the subject's appearance, keep it simple with *"a man"* or *"a woman"*. Prompts support both Chinese and English.  Use frontal, medium-to-close-up prompts for optimal results.

**Input Image:** High-resolution, frontal face images are recommended. The preprocessing pipeline handles image resolution and file types.

---

### 2.  Inference with Community LoRA:

Use `infer_with_lora.py` to use community LoRA models.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

**Recommended LoRA:** [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### 3. Video Face Swapping:

Use `infer_face_swap.py` for experimental video face swapping.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Important:**  The face swapping feature is experimental.  Adjust `denoising_strength` for the desired effect. Experiment with `--force_background_consistency` to minimize issues.

---

### 4. Infer with VACE

Use `infer_with_vace.py` for VACE compatibility.

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

You will need the VACE weights from their repository.  The `reference_video` and `reference_image` parameters are optional.  Reduce `vace_scale` for a good balance between motion and identity.  If you only use the IP image and reference video, you can reduce the weight to 0.5.

---

## Acknowledgements

We are grateful for the contributions of the following open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for dataset contributions.

---

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

---

## Contact

For questions and suggestions, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues).