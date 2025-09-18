<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly generate videos while preserving subject identity with Stand-In!</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

> **Stand-In** is a groundbreaking framework that lets you generate videos with unparalleled identity preservation. This plug-and-play solution requires training only **1%** of the base video generation model's parameters, yet achieves state-of-the-art results, surpassing methods requiring full-parameter training. Seamlessly integrate Stand-In into various tasks like subject-driven generation, pose control, video stylization, and face swapping. For more details, visit the original repository: [https://github.com/WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In)

---

## 🎉 Key Features

*   ✅ **Ultra-Efficient Training:** Train only 1% of the base model's parameters.
*   ✅ **Superior Identity Preservation:** Maintain identity consistency without sacrificing video quality.
*   ✅ **Plug-and-Play Integration:** Easily integrates with existing Text-to-Video (T2V) models.
*   ✅ **Extensive Compatibility:** Works seamlessly with community models (e.g., LoRA) and various video generation tasks.

---

## 📰 What's New
*   **[2025.08.18]** Released a version compatible with VACE, enabling pose control and other control methods, combined with Stand-In for simultaneous identity maintenance.
*   **[2025.08.16]** Updated the experimental face swapping feature.
*   **[2025.08.13]** Released the official Stand-In preprocessing ComfyUI node.
*   **[2025.08.12]** Released Stand-In v1.0 (153M parameters) with open-sourced Wan2.1-14B-T2V-adapted weights and inference code.

---

## ✨ Showcase: Examples of Stand-In in Action

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

### Discover More!

For a comprehensive gallery of results, visit our project page: [https://www.Stand-In.tech](https://www.Stand-In.tech)

---

## 🚀 Quick Start Guide

### 1. Set up your environment

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install the required packages
pip install -r requirements.txt

# (Optional) Speed up inference with Flash Attention
pip install flash-attn --no-build-isolation
```

### 2. Download Model Weights

Run the provided script to automatically download all necessary model weights into the `checkpoints` directory:

```bash
python download_models.py
```

This script downloads:

*   `wan2.1-T2V-14B` (base T2V model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (the Stand-In model)

>   *Note:* If you have the `wan2.1-T2V-14B` model already, modify `download_model.py` to skip that download, and place the model in the `checkpoints/wan2.1-T2V-14B` folder.

---

## 💡 Usage: Generating Videos with Stand-In

### Standard Inference

Use `infer.py` for identity-preserving text-to-video generation.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

*   **Prompting Tip:** Use simple prompts like *"a man"* or *"a woman"* to avoid altering the subject's facial features. Prompts support Chinese and English. The tool is designed for frontal, medium-to-close-up video generation.
*   **Input Image Recommendation:** Use a high-resolution frontal face image. The built-in preprocessing will handle various resolutions and file types.

---

### Inference with Community LoRA

Use `infer_with_lora.py` to combine Stand-In with community LoRA models for enhanced style.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend the following style LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping

Use `infer_face_swap.py` for experimental video face swapping.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

*   **Important Note:** The face swapping feature is experimental. The `denoising_strength` parameter controls the balance between background and face area redraws. Adjust it to achieve the best visual results. Higher values redraw more of the background, leading to a more natural face, but might cause contour issues.

---

### Infer with VACE for Pose-Guided Generation

Use `infer_with_vace.py` for identity-preserving video generation compatible with VACE.

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

*   **Prerequisites:** Download the corresponding weights from the `VACE` repository, or provide the `vace_path` parameter. Use `python download_models.py --vace` to download them.
*   **Input:** The input control video needs to be preprocessed using VACE's preprocessing tool. Experiment with `vace_scale` to balance motion and identity preservation. If only `ip_image` and `reference_video` are provided, you can reduce the weight to 0.5.
*   **Note:** Using Stand-In and VACE together can be more complex. Please report any issues or unexpected results.

---

## 🙏 Acknowledgements

This project builds upon these exceptional open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We extend our sincere gratitude to the authors and contributors of these projects.

We also appreciate the contribution of [Binxin Yang](https://binxinyang.github.io/) for collecting the original dataset.

---

## 📚 Citation

If you use our work, please cite our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

---

## 💬 Get in Touch

For questions and suggestions, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues). We value your feedback!