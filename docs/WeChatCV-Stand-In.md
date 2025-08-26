<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Identity-Preserving Video Generation
  </h1>

  <h3>Effortlessly create videos that maintain the identity of your subject with Stand-In, a lightweight and plug-and-play solution.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

[View the original repository](https://github.com/WeChatCV/Stand-In)

---

## ✨ Key Features of Stand-In

*   **Lightweight & Efficient:** Trains with only 1% additional parameters compared to the base model, saving valuable resources.
*   **High-Fidelity Identity:** Guarantees impressive identity consistency without compromising video generation quality.
*   **Plug-and-Play Integration:** Works seamlessly with existing Text-to-Video (T2V) models for easy adoption.
*   **Extensible & Versatile:** Compatible with LoRA and other community models, supporting various video generation tasks such as subject-driven video generation, pose-controlled video generation, video stylization, and face swapping.

---

## 📰 What's New

*   **[2025.08.18]** Released a version compatible with VACE, enabling pose control and other control methods (depth maps) combined with Stand-In for simultaneous identity preservation.
*   **[2025.08.16]** Updated the experimental face swapping feature.
*   **[2025.08.13]** Released an official Stand-In preprocessing ComfyUI node to address compatibility issues.
*   **[2025.08.12]** Launched Stand-In v1.0 (153M parameters), with open-sourced weights and inference code adapted to Wan2.1-14B-T2V.

---

## 🎬 Showcase: See Stand-In in Action!

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
### Explore More!

For additional examples and results, please visit the [Stand-In project page](https://www.stand-in.tech).

---

## 🛠️ Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference (check compatibility)
pip install flash-attn --no-build-isolation
```

### 2. Model Download

Download all required model weights automatically:

```bash
python download_models.py
```

This will download:

*   `wan2.1-T2V-14B` (base model)
*   `antelopev2` (face recognition)
*   `Stand-In` (Stand-In model)

> **Note:** If you already have the base model, modify `download_model.py` and place the model in `checkpoints/wan2.1-T2V-14B`.

---

## 🚀 Usage Instructions

### Standard Inference

Generate videos with identity preservation using the `infer.py` script:

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tips:** Use *"a man"* or *"a woman"* to avoid altering facial features. Prompts support both Chinese and English and are designed for frontal, medium-to-close-up videos.

**Input Image Recommendations:** Use a high-resolution frontal face image. Our preprocessing handles various resolutions and formats.

---

### Inference with Community LoRA

Load LoRA models alongside Stand-In:

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

Recommended LoRA: [Studio Ghibli LoRA](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping

Perform video face swapping (experimental):

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note**: Adjust `denoising_strength` for optimal face and background consistency.  `--force_background_consistency` can be used (experimentally) but may introduce artifacts.

---

### Infer with VACE

Generate videos with VACE integration:

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

Download VACE weights and specify their path.  Adjust `vace_scale` for motion and identity balance.

---

## 🤝 Acknowledgements

This project leverages these great open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)

We appreciate the contributions of their authors!

Thanks to [Binxin Yang](https://binxinyang.github.io/) for collecting the dataset.

---

## 📝 Citation

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

## 💬 Get in Touch

For questions or suggestions, please open an [issue on GitHub](https://github.com/WeChatCV/Stand-In/issues). We welcome your feedback!