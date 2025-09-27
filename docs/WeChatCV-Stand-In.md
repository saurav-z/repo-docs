<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Unlock unprecedented control over video generation with Stand-In, preserving identity seamlessly with minimal overhead.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

![Stand-In Demo](https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966)

> **Stand-In** is a groundbreaking framework that allows for **identity-preserving video generation with only a 1% increase in parameters**, outperforming methods requiring full parameter training. This plug-and-play solution seamlessly integrates with text-to-video models and supports various tasks such as subject-driven generation, pose control, video stylization, and face swapping.  Visit the original repo for more details: [WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In)

---

## Key Features

*   **Effortless Identity Preservation:** Maintain the identity of the subject in your videos with state-of-the-art face similarity results.
*   **Lightweight & Efficient:** Requires training only 1% of the parameters of the base model.
*   **Plug-and-Play Integration:** Easily integrate with existing Text-to-Video (T2V) models.
*   **Versatile Applications:** Supports diverse tasks including subject-driven generation, pose control (with VACE), video stylization, and face swapping.
*   **Community Model Compatibility:** Works with community models and LoRAs.

---

## What's New

*   **[2025.08.18]** VACE Compatibility: Now supports pose control and other control methods (e.g., depth maps) for simultaneous identity preservation.
*   **[2025.08.16]** Face Swapping Feature:  Experimental version with improvements.
*   **[2025.08.13]** ComfyUI Integration:  Official Stand-In preprocessing node released for enhanced ComfyUI experience ([Stand-In_Preprocessor_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI)).  The full official Stand-In ComfyUI node is coming soon.
*   **[2025.08.12]** Stand-In v1.0 Release:  Open-sourced `Wan2.1-14B-T2V` adapted weights and inference code (153M parameters).

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

### Explore More!
For additional examples and results, please visit: [https://stand-in-video.github.io/](https://www.Stand-In.tech)

---

## Getting Started: Quick Installation

### 1.  Set up your environment
```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference
pip install flash-attn --no-build-isolation
```

### 2. Download Necessary Models
Run the download script to automatically fetch all required models into the `checkpoints` directory.
```bash
python download_models.py
```
This will download:
*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (the Stand-In model)

> Note:  If you already have the `wan2.1-T2V-14B model` locally, you can manually edit the `download_model.py` script to comment out the relevant download code and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## Usage Guide: Video Generation with Stand-In

### Standard Inference

Generate identity-preserving videos from text prompts using `infer.py`.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompt Recommendations:**
*   For optimal results without facial alterations, use general descriptions like *"a man"* or *"a woman"*.
*   Prompts support both Chinese and English.
*   Best suited for frontal, medium-to-close-up videos.

**Input Image Best Practices:**  Use a high-resolution frontal face image.  Our built-in preprocessing pipeline handles resolution and file extensions automatically.

---

### Inference with Community LoRA

Load and apply community LoRA models with `infer_with_lora.py`.

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

### Video Face Swapping

Perform experimental video face swapping using `infer_face_swap.py`.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```
**Important Notes for Face Swapping:**

*   This feature is experimental.
*   Adjust `--denoising_strength` for optimal results. A higher value redraws more of the background, enhancing face naturalness, but may introduce minor contour issues. Lower values preserve more of the background but might result in overfitting in the face region.
*   Use `--force_background_consistency` cautiously, as it can accentuate contour problems if not correctly configured with denoising strength.

---

### Infer with VACE

Generate identity-preserving videos with pose control via VACE using the `infer_with_vace.py` script.
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
*   Download VACE weights and specify their path in the `vace_path` parameter.
```bash
python download_models.py --vace
```
*   The input control video must be preprocessed using the VACE preprocessing tool.
*   Both `reference_video` and `reference_image` are optional.
*   Adjust `vace_scale` to balance motion and identity preservation. Reducing the weight can also help with the results.
*   Using VACE with Stand-In is more complex; please report any issues.

---

## To-Do List

*   \[x] Release IP2V inference script (compatible with community LoRA).
*   \[x] Open-source model weights compatible with Wan2.1-14B-T2V: `Stand-In_Wan2.1-T2V-14B_153M_v1.0`.
*   \[ ] Open-source model weights compatible with Wan2.2-T2V-A14B.
*   \[ ] Release training dataset, data preprocessing scripts, and training code.

---

## Acknowledgements

This project leverages the following open-source projects:
*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We are grateful to the authors and contributors of these projects.

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for contributing to our dataset.

---

## Citation

If you find our work helpful, please cite our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

---

## Contact Us

For questions and feedback, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues).