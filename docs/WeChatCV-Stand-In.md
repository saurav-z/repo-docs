<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Effortlessly Preserve Identity in Your Videos
  </h1>

  <h3>A Lightweight and Plug-and-Play Identity Control for Video Generation</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

[**Explore the full potential of Stand-In and revolutionize your video creation!**](https://github.com/WeChatCV/Stand-In)

---

## ✨ Key Features of Stand-In

*   **Minimal Overhead:** Trains with only 1% additional parameters, minimizing resource usage.
*   **Exceptional Fidelity:** Achieves state-of-the-art results in both Face Similarity and Naturalness, preserving identity beautifully.
*   **Easy Integration:** "Plug-and-Play" design makes it simple to add to your existing Text-to-Video (T2V) workflows.
*   **Versatile Applications:** Supports a wide range of tasks, including:
    *   Text-to-Video Generation
    *   Non-Human Subject-Preserving Video Generation
    *   Video Stylization with LoRA
    *   Video Face Swapping
    *   Pose-Guided Video Generation (with VACE)

---

## 🚀 Recent Updates

*   **[2025.08.18]** Released a version compatible with VACE, enabling pose control and other control methods while maintaining identity.
*   **[2025.08.16]** Updated the experimental face swapping feature.
*   **[2025.08.13]** Released the official Stand-In preprocessing ComfyUI node.
*   **[2025.08.12]** Launched Stand-In v1.0 (153M parameters), with open-sourced weights and inference code.

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

### For more results, please visit [https://stand-in-video.github.io/](https://www.Stand-In.tech)

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

# (Optional) Install Flash Attention for faster inference.  Ensure compatibility with your GPU and CUDA version.
pip install flash-attn --no-build-isolation
```

### 2. Model Download

Download the necessary models using the provided script:

```bash
python download_models.py
```

This script automatically downloads:
*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

>   **Note:** If you have the base model already, you can manually adjust the `download_model.py` script to skip downloading and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## 🎬 Usage: Inference and Beyond

### Standard Inference

Use `infer.py` for basic identity-preserving text-to-video generation.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tips:** To preserve facial features, use general terms like *"a man"* or *"a woman"* without further appearance descriptions. Prompts support both English and Chinese. Focus prompts on generating frontal, medium-to-close-up videos.

**Input Image Recommendation:** Use a high-resolution, frontal face image for the best results. Our built-in preprocessing handles various resolutions and file types.

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

We recommend using the stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping

Experiment with video face swapping using `infer_face_swap.py`:

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Important Face Swap Notes:**  Face swapping is experimental due to limitations in the base model.

Adjust `denoising_strength` to control background and face area changes. Higher values redraw the background more, enhancing face naturalness, but may introduce artifacts. Lower values preserve the background but can lead to overfitting in the face area.  Consider `--force_background_consistency` cautiously; it can cause contour issues.

### Infer with VACE
Use the `infer_with_vace.py` script to perform identity-preserving video generation with Stand-In, compatible with VACE.
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
You need to download the corresponding weights from the `VACE` repository or provide the path to the `VACE` weights in the `vace_path` parameter.

```bash
python download_models.py --vace
```

The input control video needs to be preprocessed using VACE's preprocessing tool. Both `reference_video` and `reference_image` are optional and can exist simultaneously. Additionally, VACE’s control has a preset bias towards faces, which affects identity preservation. Please lower the `vace_scale` to a balance point where both motion and identity are preserved. When only `ip_image` and `reference_video` are provided, the weight can be reduced to 0.5.

Using both Stand-In and VACE together is more challenging than using Stand-In alone. We are still maintaining this feature, so if you encounter unexpected outputs or have other questions, feel free to raise them in the issue.

---

## 🙏 Acknowledgements

We are deeply grateful to the following open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

A special thank you to [Binxin Yang](https://binxinyang.github.io/) for their contribution to the dataset.

---

## 📚 Citation

If our work has helped your research, please cite us:

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

For any questions or suggestions, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues).  We value your feedback!