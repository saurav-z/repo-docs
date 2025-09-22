<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Identity-Preserving Video Generation
  </h1>

  <h3>Generate high-quality videos while maintaining subject identity with minimal training!</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

[View the original repository on GitHub](https://github.com/WeChatCV/Stand-In)

---

## Stand-In: Revolutionizing Video Generation with Lightweight Identity Control

**Stand-In** is a groundbreaking, plug-and-play framework that empowers users to create stunning videos while preserving the identity of the subject with unparalleled efficiency.  By training only **1%** of additional parameters, Stand-In achieves state-of-the-art results in both Face Similarity and Naturalness, surpassing methods that require full-parameter training.  Seamlessly integrate Stand-In with other tasks, including subject-driven generation, pose control, video stylization, and face swapping.

---

## 🎉 Key Features

*   **Lightweight & Efficient:** Train with only 1% additional parameters of the base model.
*   **Superior Identity Preservation:** Maintain strong facial identity without compromising video quality.
*   **Plug-and-Play Integration:** Easily integrate into existing text-to-video (T2V) models.
*   **Extensible & Versatile:** Compatible with LoRA and various downstream video tasks (e.g., stylization, face swap).
*   **Community Ready:** Fully compatible with custom ComfyUI nodes.

---

## 🚀 Showcase: Witness the Power of Stand-In

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
### Explore More: Discover the Full Potential

For additional video examples and in-depth demonstrations, please visit our project page: [https://www.Stand-In.tech](https://www.Stand-In.tech)

---

## 🛠️ Getting Started: Quick Setup Guide

### 1.  Environment Setup
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

### 2.  Model Download
Utilize our automatic download script to effortlessly acquire all the necessary model weights and place them in the `checkpoints` directory:

```bash
python download_models.py
```

This script automatically downloads the following models:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

>   **Important:** If you already possess the `wan2.1-T2V-14B` model locally, modify the `download_model.py` script to comment out the corresponding download code and ensure the model is placed within the `checkpoints/wan2.1-T2V-14B` directory.

---

## ⚙️ Usage: Dive into Video Generation

### Standard Inference

Generate identity-preserving videos using the `infer.py` script.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompt Guidance:** Achieve optimal results by using prompts such as *"a man"* or *"a woman"* to avoid altering facial features. Both Chinese and English prompts are supported. For best results, aim for frontal, medium-to-close-up videos.

**Input Image Recommendations:** For superior results, provide a high-resolution frontal face image.  Our built-in preprocessing pipeline handles resolution and file extensions automatically.

---

### Inference with Community LoRA

Combine Stand-In with community LoRA models for enhanced stylization using the `infer_with_lora.py` script.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend this stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping

Experiment with face swapping using the `infer_face_swap.py` script.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Important Note:** Face swapping is an experimental feature within our framework.

The `denoising_strength` parameter regulates the degree of background and face area modification:

*   Higher `denoising_strength`: Greater background redraw, potentially more natural face.
*   Lower `denoising_strength`: Reduced background redraw, possible overfitting in the face.

Enable `--force_background_consistency` for complete background consistency, but be aware of potential contour issues. Experiment with different `denoising_strength` values for the best results.  If minor background changes are acceptable, avoid enabling this feature.

---

### Infer with VACE

Integrate Stand-In with VACE for advanced control over video generation, using `infer_with_vace.py`.

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

Ensure that you've downloaded the corresponding weights from the `VACE` repository or provide the `VACE` weights' path through the `vace_path` parameter.

```bash
python download_models.py --vace
```

The input control video must be preprocessed using the VACE preprocessing tool. Both `reference_video` and `reference_image` are optional and can be used in conjunction. Note that VACE's control incorporates a bias towards faces, which impacts identity preservation. Adjust `vace_scale` to achieve an equilibrium between motion control and identity preservation. Reduce the weight to 0.5 when only `ip_image` and `reference_video` are provided.

Combining Stand-In and VACE presents greater challenges compared to using Stand-In alone. Please don't hesitate to raise any questions or issues you encounter.

---

## 🤝 Acknowledgements

This project leverages the following exceptional open-source resources:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We express our sincere gratitude to the creators and contributors of these invaluable projects.

The original raw material of our dataset was collected with the help of our team member [Binxin Yang](https://binxinyang.github.io/), and we appreciate his contribution!

---

## ✏ Citation

If you find Stand-In helpful for your research, please cite our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

---

## 📬 Contact Us

For inquiries and suggestions, connect with us through [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues). We look forward to your feedback!