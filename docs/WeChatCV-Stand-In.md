<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly preserve identities in your videos with Stand-In, a plug-and-play solution that's easy to use.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

**[Visit the original Stand-In repository](https://github.com/WeChatCV/Stand-In)**

---

## Key Features:

*   **Minimal Overhead:** Only requires training 1% additional parameters compared to the base model.
*   **Exceptional Quality:** Achieves state-of-the-art results in face similarity and naturalness.
*   **Simple Integration:** A plug-and-play solution that seamlessly integrates into existing Text-to-Video (T2V) models.
*   **Versatile Compatibility:** Works with LoRA models and supports various video generation tasks (subject-driven, pose-controlled, stylization, and face swapping).

---

## What's New

*   **[2025.08.18]** Released VACE compatibility for pose control and other control methods.
*   **[2025.08.16]** Updated the experimental face swapping feature.
*   **[2025.08.13]** Released official Stand-In preprocessing ComfyUI node.
*   **[2025.08.12]** Released Stand-In v1.0 (153M parameters) and open-sourced weights and inference code.

---

## Showcase: Impressive Results

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)| "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3) |
|![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)| "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." |![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab)|

---

### Non-Human Subject Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" />|"A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads."|![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5) |

---

### Stylized Video Generation

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

### For More Results

Explore more examples and see Stand-In in action at [https://stand-in-video.github.io/](https://www.Stand-In.tech).

---

## Getting Started: Quick Setup Guide

### 1.  Environment Setup:

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install required packages
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference
pip install flash-attn --no-build-isolation
```

### 2.  Model Download:

Download all necessary models using the provided script:

```bash
python download_models.py
```

This script downloads:

*   `wan2.1-T2V-14B` (base model)
*   `antelopev2` (face recognition)
*   `Stand-In` (the Stand-In model)

> **Important:** If you have `wan2.1-T2V-14B` already, you can modify `download_model.py` to skip its download and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## Usage: Core Scripts

### Standard Inference

Use the `infer.py` script:

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tips:** Use *"a man"* or *"a woman"* to avoid modifying facial features. The prompt is meant for frontal, medium-to-close-up videos.

**Input Recommendations:** Use a high-resolution frontal face image. Our preprocessing pipeline handles various resolutions and file types.

---

### Inference with Community LoRA

Use the `infer_with_lora.py` script:

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

**Example LoRA:**  [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping (Experimental)

Use the `infer_face_swap.py` script:

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Important Note:**  Face swapping is experimental. Adjust `denoising_strength` for optimal results. Use `--force_background_consistency` cautiously.

---

### Inference with VACE

Use the `infer_with_vace.py` script:

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

**Prerequisites:** Download VACE weights and preprocess the control video using VACE's preprocessing tool. Adjust `vace_scale` for balance between motion and identity.

---

## To-Do List

*   \[x] Release IP2V inference script.
*   \[x] Open-source model weights for Wan2.1-14B-T2V: `Stand-In_Wan2.1-T2V-14B_153M_v1.0`.
*   \[ ] Open-source model weights for Wan2.2-T2V-A14B.
*   \[ ] Release training dataset, scripts, and code.

---

## Acknowledgements

We are grateful for the contributions of the following open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base model)

Thanks to [Binxin Yang](https://binxinyang.github.io/) for collecting dataset materials!

---

## Citation

If our work is helpful, please cite:

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

For any questions or suggestions, reach out via [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues).