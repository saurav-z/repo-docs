<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly preserve identities in your videos with Stand-In, a plug-and-play solution.</h3>

  [![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
  [![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
  [![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

  [View the original repository](https://github.com/WeChatCV/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

---

## Key Features

*   ✅ **Lightweight & Efficient:** Achieves state-of-the-art identity preservation with only 1% additional trainable parameters compared to the base video generation model.
*   ✅ **High-Fidelity Results:** Delivers outstanding identity consistency without sacrificing video generation quality.
*   ✅ **Plug-and-Play Integration:** Seamlessly integrates with existing text-to-video (T2V) models, making it easy to use.
*   ✅ **Extensible & Versatile:** Compatible with community models like LoRA and supports various downstream video tasks, including face swapping, and pose-guided generation.

---

## What's New

*   **[2025.08.18]** Compatibility with VACE released. Added pose control and other control methods such as depth maps.
*   **[2025.08.16]** Updated the experimental face swapping feature.
*   **[2025.08.13]** Official Stand-In preprocessing ComfyUI node released: [https://github.com/WeChatCV/Stand-In\_Preprocessor\_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI).
*   **[2025.08.12]** Released Stand-In v1.0 (153M parameters).

---

## Showcase: See Stand-In in Action!

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt                                                                                                                                                                                                                                                                                                         | Generated Video                                                                                                                                                                               |
| :-------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![](<https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541>) | "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." | ![](<https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3>)                                                                                                  |
| ![](<https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172>) | "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind."                     | ![](<https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab>)                                                                                                  |

---

### Non-Human Subjects-Preserving Video Generation

| Reference Image                                                                                                 | Prompt                                                                                                                                   | Generated Video                                                                                                                                                                       |
| :-------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" /> | "A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads." | ![](<https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5>)                                                                                                  |

---

### Identity-Preserving Stylized Video Generation

| Reference Image                                                                                                 | LoRA           | Generated Video                                                                                                                                                                  |
| :-------------------------------------------------------------------------------------------------------------- | :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![](<https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1>) | Ghibli LoRA    | ![](<https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f>)                                                                                                  |

---

### Video Face Swapping

| Reference Video                                                                                                 | Identity                                                                                                 | Generated Video                                                                                                                                                                |
| :-------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![](<https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701>) | <img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" /> | ![](<https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50>)                                                                                                 |

---

### Pose-Guided Video Generation (With VACE)

| Reference Pose                                                                                                | First Frame                                                                                               | Generated Video                                                                                                                                                               |
| :------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ![](<https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94>) | <img width="719" height="415" alt="Image" src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" /> | ![](<https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44>)                                                                                                 |

---

For more examples and results, visit the project website: [https://www.Stand-In.tech](https://www.Stand-In.tech)

---

## Getting Started

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

### 2. Model Download

Automate your setup with our convenient download script:

```bash
python download_models.py
```

This script will automatically download:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

> **Note:** If you have the `wan2.1-T2V-14B model` locally, modify the `download_model.py` script to skip that download and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## Usage Guide

### Standard Inference

Generate videos with identity preservation using the `infer.py` script.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

*   **Prompting Tip:** For consistent facial features, use prompts like *"a man"* or *"a woman"* without adding details about appearance.  Prompts support English and Chinese. The prompt is intended for generating frontal, medium-to-close-up videos.
*   **Input Image:**  Use high-resolution frontal face images for the best results. Our preprocessing handles different resolutions and file types.

### Inference with Community LoRA

Load community LoRA models with `infer_with_lora.py`.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend trying the Ghibli LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping

Experiment with face swapping using the `infer_face_swap.py` script.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note:** Face swapping is experimental due to limitations with Wan2.1.

*   Adjust `--denoising_strength` for optimal face and background blending.
*   Use `--force_background_consistency` with caution, as it may introduce artifacts.

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

## Acknowledgements

This project utilizes and builds upon the following open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We are immensely grateful to the creators and contributors of these projects.

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for their contributions to the dataset.

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

## Get in Touch

For any questions or suggestions, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues).
We value your feedback!