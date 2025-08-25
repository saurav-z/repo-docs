<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Generate stunning videos that preserve identity with Stand-In, a plug-and-play solution!</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

![Stand-In Example](https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966)

**[View the original repository](https://github.com/WeChatCV/Stand-In)**

---

## **Stand-In: Key Features & Benefits**

Stand-In is a cutting-edge framework for generating videos while perfectly preserving the identity of the subject. This innovative approach achieves state-of-the-art results with minimal computational overhead.

*   ✅ **Efficient Training:** Train only **1%** of the base model's parameters.
*   ✅ **Exceptional Fidelity:**  Maintains strong identity consistency without sacrificing video generation quality.
*   ✅ **Plug-and-Play Integration:** Seamlessly integrates with existing Text-to-Video (T2V) models.
*   ✅ **Highly Versatile:**  Compatible with community models (e.g., LoRA) and supports various downstream video tasks, including subject-driven, pose-controlled, stylized, and face-swapped video generation.

---

## **What's New**

Stay updated on the latest developments:

*   **[2025.08.18]** Released VACE compatibility for advanced control capabilities.
*   **[2025.08.16]** Experimental face swapping feature updated.
*   **[2025.08.13]** Official Stand-In Preprocessing ComfyUI node released.
*   **[2025.08.12]** Stand-In v1.0 (153M parameters) with open-sourced weights and inference code.

---

## **Showcase:  See Stand-In in Action!**

Stand-In excels in various video generation scenarios:

### **Identity-Preserving Text-to-Video Generation**

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)| "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3) |
|![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)| "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." |![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab)|

---

### **Non-Human Subjects-Preserving Video Generation**

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" />|"A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads."|![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5) |

---

### **Identity-Preserving Stylized Video Generation**

| Reference Image | LoRA | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1)|Ghibli LoRA|![Image](https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f)|
---

### **Video Face Swapping**

| Reference Video | Identity | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701)|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" />|![Image](https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50)|

---

### **Pose-Guided Video Generation (With VACE)**

| Reference Pose | First Frame | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94)|<img width="719" height="415" alt="Image" src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" />|![Image](https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44)|

---

### **Explore More Examples**
Visit [https://www.Stand-In.tech](https://www.Stand-In.tech) for more impressive results!

---

## **Getting Started: Quick Start Guide**

Follow these steps to get Stand-In up and running:

### 1.  Environment Setup

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install required packages
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference (ensure compatibility)
pip install flash-attn --no-build-isolation
```

### 2. Model Download
Easily download all necessary model weights using our automated script.

```bash
python download_models.py
```

This script will download the following models:
* `wan2.1-T2V-14B` (base text-to-video model)
* `antelopev2` (face recognition model)
* `Stand-In` (our Stand-In model)

> **Note:** If you already have the `wan2.1-T2V-14B` model, modify the `download_model.py` script to prevent redownloading and place the model in the specified `checkpoints` directory.

---

## **Usage:  Generating Videos with Stand-In**

### **Standard Inference**

Run the `infer.py` script for identity-preserving text-to-video generation.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tips:**  Use *"a man"* or *"a woman"* for simple appearance descriptions. Prompts can be in Chinese or English, and are optimized for frontal, medium-to-close-up videos.

**Input Image Best Practices:**  Use a high-resolution, frontal face image for the best results. Our built-in preprocessing handles various resolutions and file types.

---

### **Inference with Community LoRA**

Load and use community LoRA models alongside Stand-In with the `infer_with_lora.py` script.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

Recommended LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### **Video Face Swapping (Experimental)**

Utilize the `infer_face_swap.py` script for video face swapping.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Important Notes:**  Face swapping is experimental. Adjust `denoising_strength` for optimal results.  Higher values redraw more background and can yield a more natural face.  `--force_background_consistency` maintains background consistency but may cause contour issues.

---

### **Infer with VACE**

Generate videos using Stand-In with VACE for advanced control.

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

*   Download VACE weights from the VACE repository and provide the path.
*   Preprocess the input control video using VACE's preprocessing tool.
*   Adjust `vace_scale` to balance motion and identity preservation.

---

## **Acknowledgements**

We are deeply grateful to the following open-source projects that made this work possible:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for the original dataset collection!

---

## **Citation**

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

## **Get in Touch**

For questions or suggestions, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues). We value your feedback!