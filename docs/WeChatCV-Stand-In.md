<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Identity-Preserving Video Generation Made Easy
  </h1>

  <h3>Generate stunning videos while preserving the identity of your subjects with Stand-In, a lightweight and plug-and-play solution.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<p align="center">
  <a href="https://github.com/WeChatCV/Stand-In">View the original repository</a>
</p>

---

**Stand-In** offers a revolutionary approach to video generation, preserving the identity of subjects with remarkable efficiency. By training only **1%** of additional parameters on top of the base video generation model, Stand-In achieves state-of-the-art results in both face similarity and naturalness, outperforming methods requiring full-parameter training. This makes Stand-In the ideal solution for a variety of applications, including subject-driven and pose-controlled video generation, video stylization, and even face swapping.

## 🚀 Key Features

*   ✅ **Lightweight Training:** Train only 1% additional parameters.
*   ✅ **High Fidelity:** Excellent identity preservation without sacrificing video quality.
*   ✅ **Plug-and-Play Integration:** Easy to integrate into existing T2V (Text-to-Video) models and workflows.
*   ✅ **Versatile Compatibility:** Supports community models like LoRA and other downstream video tasks.
*   ✅ **Experimental Face Swap:**  Provides preliminary functionality for face swapping in videos.

## ✨ Showcase: See Stand-In in Action!

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

### Video Face Swapping (Experimental)

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
For additional examples and a deeper dive into the capabilities of Stand-In, please visit the project's dedicated page: [https://www.Stand-In.tech](https://www.Stand-In.tech)

## 📰 Recent Updates

*   **[2025.08.18]** Compatibility with VACE released, including pose control.
*   **[2025.08.16]** Face swapping feature updated.
*   **[2025.08.13]** Official Stand-In preprocessor ComfyUI node released.
*   **[2025.08.12]** Stand-In v1.0 released, with Wan2.1-14B-T2V adapted weights and inference code open-sourced.

## ⚙️ Getting Started: Quick Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/WeChatCV/Stand-In.git
    cd Stand-In
    ```

2.  **Set up a Conda Environment:**

    ```bash
    conda create -n Stand-In python=3.11 -y
    conda activate Stand-In
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Install Flash Attention (for faster inference):**

    ```bash
    pip install flash-attn --no-build-isolation
    ```

    *Ensure your GPU and CUDA versions are compatible with Flash Attention.*

5.  **Download Models:**

    ```bash
    python download_models.py
    ```

    *This will download the necessary models, including wan2.1-T2V-14B, antelopev2, and Stand-In, into the `checkpoints` directory.*

## 💻 Usage Guide

### Standard Inference

Generate videos using text prompts and identity images.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

*   **Prompt Tips:** Use simple prompts like *"a man"* or *"a woman"* to avoid altering facial features.  Both Chinese and English prompts are supported.  Focus on frontal, medium-to-close-up shots.
*   **Input Image Recommendation:** High-resolution, frontal face images yield the best results.

### Inference with Community LoRA

Combine Stand-In with community LoRA models for enhanced stylization.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

*   **Recommended LoRA:** [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping (Experimental)

Perform video face swapping with the provided script.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

*   **Note:** The face swapping feature is experimental. Adjust `denoising_strength` to balance background redraw and face realism.  Consider `--force_background_consistency` with caution due to potential contour issues.

### Infer with VACE

Generate videos with pose control using VACE.

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

*   Download VACE weights or provide the path to them.
*   Preprocess input videos using VACE's preprocessing tools.
*   Adjust `vace_scale` for optimal balance between motion and identity.

```bash
python download_models.py --vace
```

## 🤝 Acknowledgements

This project is built upon and inspired by several outstanding open-source contributions:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)

A special thanks to [Binxin Yang](https://binxinyang.github.io/) for their contributions to the dataset.

## 📚 Citation

If you find our work useful, please cite our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

## 💬 Contact Us

For any questions or suggestions, please reach out via [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues). We look forward to your feedback!