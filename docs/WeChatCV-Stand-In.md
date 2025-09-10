<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly create videos while preserving the identity of the subject, with just 1% extra parameters!</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)
</div>

[**Check out the original repository here!**](https://github.com/WeChatCV/Stand-In)

---

## Overview

**Stand-In** is a revolutionary, plug-and-play framework designed to provide robust identity control for video generation.  By training only a minimal number of parameters, approximately **1%** of the base model, Stand-In achieves state-of-the-art results in both Face Similarity and Naturalness, surpassing methods that require full parameter training.  Stand-In seamlessly integrates with various video generation tasks, including subject-driven generation, pose-controlled generation, video stylization, and face swapping.

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

## ✨ Key Features

*   **Minimal Parameter Training:** Achieves outstanding results with only 1% of the base model's parameters.
*   **Superior Fidelity:** Maintains exceptional identity consistency without compromising video quality.
*   **Easy Integration:**  Plug-and-play functionality makes it simple to integrate with existing text-to-video (T2V) models.
*   **Broad Compatibility:** Works seamlessly with community models (LoRA) and supports various downstream video tasks.

## 🚀 What's New

*   **[2025.08.18]**  VACE compatibility released!  Explore pose control and other control methods like depth maps, all while maintaining identity.
*   **[2025.08.16]**  Experimental face swapping feature updated!
*   **[2025.08.13]** Official Stand-In preprocessing ComfyUI node released: [https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI)
*   **[2025.08.12]**  Stand-In v1.0 (153M parameters) released, along with open-sourced Wan2.1-14B-T2V-adapted weights and inference code.

## 🌟 Showcase: See Stand-In in Action

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)| "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3) |
|![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)| "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." |![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab)|

---

### Non-Human Subject-Preserving Video Generation

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

### Explore More!

For more impressive examples, visit our project page at [https://www.Stand-In.tech](https://www.stand-in-video.github.io/)

## 🛠️ Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention (recommended for faster inference)
# Note: Ensure your GPU and CUDA version are compatible.
pip install flash-attn --no-build-isolation
```

### 2. Model Download

Use the provided script to automatically download all required model weights:

```bash
python download_models.py
```

This script downloads:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

>   **Note:**  If you have the `wan2.1-T2V-14B` model locally, you can modify the `download_models.py` script to skip the download and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

## 💻 Usage

### Standard Inference

Generate identity-preserving videos using the `infer.py` script:

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

*   **Prompt Tip:**  For consistent facial features, use simple prompts like *"a man"* or *"a woman"*.  Supports both English and Chinese.
*   **Input Image:** A high-resolution, frontal face image is recommended for optimal results.  Our preprocessing pipeline handles various resolutions and file types.

### Inference with Community LoRA

Load community LoRA models alongside Stand-In using the `infer_with_lora.py` script:

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend this stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping (Experimental)

Experiment with face swapping using the `infer_face_swap.py` script:

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

*   **Note:** Face swapping is experimental. Adjust `--denoising_strength` to control the balance between background consistency and face naturalness.

### Infer with VACE

Generate videos with pose control using VACE and Stand-In via the `infer_with_vace.py` script:

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

*   Download the VACE weights from their repository and specify the path in the `vace_path` parameter.
*   Preprocess your input control video using VACE's preprocessing tool.  `reference_video` and `reference_image` are both optional.
*   Adjust the `vace_scale` to balance motion and identity preservation.

## 🤝 Acknowledgements

This project is built upon the following excellent open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We are grateful to the authors and contributors of these projects.

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for collecting the dataset's original material.

## ✏️ Citation

If you find our work useful, please cite our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

## 📬 Contact

For any questions or suggestions, please open an [Issue](https://github.com/WeChatCV/Stand-In/issues) on GitHub.  We welcome your feedback!