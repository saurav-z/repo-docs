<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly maintain identity in your videos with Stand-In, a plug-and-play solution that only requires training on 1% of the base model's parameters.</h3>

  [![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
  [![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
  [![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

  [View the original repository on GitHub](https://github.com/WeChatCV/Stand-In)

</div>

<img width="5333" height="2983" alt="Stand-In Example" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

---

## Key Features

*   **Lightweight and Efficient:** Train only 1% of the base model's parameters.
*   **State-of-the-Art Results:** Achieve superior Face Similarity and Naturalness.
*   **Plug-and-Play Integration:** Seamlessly integrates with existing text-to-video models.
*   **Versatile Applications:** Supports subject-driven, pose-controlled video generation, video stylization, and face swapping.

---

## What's New

*   **[2025.08.18]** VACE Compatibility:  Now supports pose control and other control methods like depth maps combined with Stand-In for simultaneous identity preservation.
*   **[2025.08.16]** Face Swapping Feature Update:  An experimental face swapping feature is now available.
*   **[2025.08.13]** Official Preprocessing Node for ComfyUI:  Addressing issues with a community implementation, the official preprocessing node for ComfyUI is released (https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI).  Full Stand-In ComfyUI node is forthcoming.
*   **[2025.08.12]** Stand-In v1.0 Release:  Open-sourced weights and inference code adapted for Wan2.1-14B-T2V (153M parameters).

---

## Showcase

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt                                                                                                                                                                                                                                                                | Generated Video                                                                                                 |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------: |
|      ![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)      | "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." | ![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3) |
|     ![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)      | "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind."                        |  ![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab) |

---

### Non-Human Subjects-Preserving Video Generation

| Reference Image                                                                                                                                                                                                                             | Prompt                                                                                                                                                              | Generated Video                                                                                              |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: |
| <img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" /> | "A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads." | ![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5) |

---

### Identity-Preserving Stylized Video Generation

| Reference Image                                                                                                                                                                                                                            | LoRA        | Generated Video                                                                                             |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------: | :----------------------------------------------------------------------------------------------------------: |
| <img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1" /> | Ghibli LoRA | ![Image](https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f) |

---

### Video Face Swapping

| Reference Video                                                                                                                                                                                                                             | Identity                                                                                                                                                              | Generated Video                                                                                             |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: |
|  ![Image](https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701) |  <img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" />  |  ![Image](https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50)  |

---

### Pose-Guided Video Generation (With VACE)

| Reference Pose                                                                                                                                                                                                                             | First Frame                                                                                                                                                               | Generated Video                                                                                             |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: |
|  ![Image](https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94) |  <img width="719" height="415" alt="Image" src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" /> |  ![Image](https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44) |

---

For more examples, visit the project page: [https://stand-in-video.github.io/](https://www.Stand-In.tech)

---

## Getting Started

### 1. Environment Setup

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

Run the automatic download script to retrieve all necessary model weights.  These will be placed in the `checkpoints` directory.

```bash
python download_models.py
```

This script downloads:
*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

> **Note:** If you have the `wan2.1-T2V-14B` model already, modify `download_model.py` to comment out the relevant download and manually place the model in `checkpoints/wan2.1-T2V-14B`.

---

## Usage

### Standard Inference

Generate videos with identity preservation using the `infer.py` script.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting Tip:**  For minimal facial alteration, use *"a man"* or *"a woman"* in your prompt, avoiding detailed descriptions. Supports Chinese and English. This prompt style is designed for frontal, medium-to-close-up videos.

**Input Image Recommendation:**  Use a high-resolution frontal face image. Our preprocessing pipeline handles different resolutions and file types automatically.

---

### Inference with Community LoRA

Load and use community LoRA models alongside Stand-In with the `infer_with_lora.py` script.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

Recommended LoRA:  [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping

Experiment with video face swapping using the `infer_face_swap.py` script.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note:**  Face swapping is experimental, as Wan2.1 lacks inpainting capabilities. Adjust `denoising_strength` to fine-tune background and face consistency.  Higher values redraw more of the background, potentially creating more natural faces.  Lower values may result in overfitting in the face area.  `--force_background_consistency` can force background consistency at the potential cost of minor contour issues.

---

### Infer with VACE

Use the `infer_with_vace.py` script for pose-guided video generation with Stand-In, compatible with VACE.
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

This project leverages the following open-source projects:
*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We extend our gratitude to the authors and contributors of these projects.

We also thank [Binxin Yang](https://binxinyang.github.io/) for their help in the original dataset collection.

---

## Citation

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

## Contact

For questions or suggestions, please open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues).  We welcome your feedback!