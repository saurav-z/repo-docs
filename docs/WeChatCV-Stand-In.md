<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly generate videos that preserve identity with minimal training!</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

---

## Stand-In: Preserve Identities in Your Videos

**Stand-In** is a cutting-edge, plug-and-play framework designed to revolutionize video generation by ensuring consistent identities. By training only a tiny fraction of additional parameters (just 1%!), Stand-In achieves state-of-the-art results in both Face Similarity and Naturalness, surpassing methods requiring full-parameter training.  Want to know more?  Explore the original repo [here](https://github.com/WeChatCV/Stand-In).

**Key Features:**

*   ✅ **Efficient Training:** Train with only 1% additional parameters.
*   ✅ **High Fidelity:** Maintain outstanding identity consistency.
*   ✅ **Plug-and-Play:** Seamlessly integrates into existing T2V (Text-to-Video) models.
*   ✅ **Extensible:** Compatible with LoRA and various downstream video tasks.

---

## 🌟 Showcase: See Stand-In in Action!

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

For more stunning examples and detailed results, visit our project page: [https://www.Stand-In.tech](https://www.stand-in.tech/)

---

## 📰 Latest News

*   **[2025.08.18]** VACE compatibility released!  Now supports pose control and other methods alongside Stand-In.
*   **[2025.08.16]** Updated experimental face swapping feature!
*   **[2025.08.13]**  Official Stand-In preprocessor ComfyUI node released.
*   **[2025.08.12]** Stand-In v1.0 (153M parameters) released, along with Wan2.1-14B-T2V-adapted weights and inference code.

---

## 🚀 Quick Start: Get Up and Running!

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

# (Optional) Install Flash Attention for faster inference
pip install flash-attn --no-build-isolation
```

### 2. Model Download

```bash
python download_models.py
```

This script downloads:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

> **Note:**  If you have `wan2.1-T2V-14B` already, modify `download_model.py`.

---

## 🧪 Usage: Generate Your Videos!

### Standard Inference

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompt Tips:** Avoid detailed descriptions of the subject's appearance (e.g., just *"a man"*).  Prompts support both English and Chinese. Ideal for frontal, medium-to-close-up shots.

**Input Image:** Use high-resolution frontal face images for best results.

### Inference with Community LoRA

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

**Recommended LoRA:** [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Important Face Swap Note:** This feature is experimental. Adjust `--denoising_strength` to balance background redraw and face naturalness.  Use `--force_background_consistency` with caution.

### Infer with VACE

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

Download VACE weights or provide the path via `--vace_path`.

```bash
python download_models.py --vace
```

Preprocess the input control video using VACE's tools. Adjust `--vace_scale` to balance motion and identity preservation.  Contact us with any questions.

---

## 🤝 Acknowledgements

Stand-In builds upon the work of:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)

Thank you to the creators and contributors!

Dataset material was collected with the help of [Binxin Yang](https://binxinyang.github.io/)

---

## ✏ Citation

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

Got questions or suggestions? Reach out via [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues) !