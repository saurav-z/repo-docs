<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly preserve identities in your videos with Stand-In, a plug-and-play solution that enhances video generation models with minimal training.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

---

## Key Features

*   🎯 **Identity Preservation:** Maintain the identity of subjects in generated videos with remarkable accuracy.
*   ⚡ **Lightweight & Efficient:** Train only 1% additional parameters compared to the base model for state-of-the-art results.
*   🔌 **Plug-and-Play Integration:** Seamlessly integrates with existing text-to-video (T2V) models for easy use.
*   🚀 **Versatile Applications:** Supports subject-driven, pose-controlled video generation, video stylization, and face swapping.
*   🎨 **Community Compatible:** Works with LoRA and other community models for expanded creative possibilities.

---

## News & Updates

*   **[2025.08.18]** VACE Compatibility Released! Now you can combine pose control and depth maps with Stand-In to maintain identity.
*   **[2025.08.16]** Updated experimental Face Swapping feature available for testing.
*   **[2025.08.13]** Official Stand-In preprocessing ComfyUI node released to address compatibility issues.  (See: [https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI))
*   **[2025.08.12]** Stand-In v1.0 (153M parameters) released, with Wan2.1-14B-T2V adapted weights and inference code now open-sourced.

---

## Showcase

### Identity-Preserving Text-to-Video Generation

| Reference Image                                                                                               | Prompt                                                                                                                                                                                                                                                      | Generated Video                                                                                                 |
| :------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541" width="200"/>   | "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." | <img src="https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3" width="200"/>   |
| <img src="https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172" width="200"/>   | "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind."                                 | <img src="https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab" width="200"/>   |

---

### Non-Human Subjects-Preserving Video Generation

| Reference Image                                                                                                 | Prompt                                                                                                                                                           | Generated Video                                                                                                  |
| :-------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" width="200"/> | "A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads." | <img src="https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5" width="200"/>  |

---

### Identity-Preserving Stylized Video Generation

| Reference Image                                                                                                 | LoRA          | Generated Video                                                                                                   |
| :-------------------------------------------------------------------------------------------------------------- | :------------ | :---------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1" width="200"/> | Ghibli LoRA | <img src="https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f" width="200"/> |

---

### Video Face Swapping

| Reference Video                                                                                                 | Identity                                                                                                             | Generated Video                                                                                                  |
| :-------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701" width="200"/> | <img src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" width="200"/>         | <img src="https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50" width="200"/>  |

---

### Pose-Guided Video Generation (With VACE)

| Reference Pose                                                                                                 | First Frame                                                                                                           | Generated Video                                                                                                  |
| :-------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94" width="200"/> | <img src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" width="200"/>         | <img src="https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44" width="200"/>  |

---

For more results, visit our project page: [https://stand-in-video.github.io/](https://www.Stand-In.tech)

---

## Getting Started

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

Run the download script to automatically fetch all required models into the `checkpoints` directory:

```bash
python download_models.py
```

This downloads:

*   `wan2.1-T2V-14B` (base model)
*   `antelopev2` (face recognition)
*   `Stand-In` (our model)

>   **Note:** If you have the base model already, modify the `download_model.py` script to comment out the relevant download and place the model in the appropriate directory.

---

## Usage

### Standard Inference

Generate videos with identity preservation using `infer.py`:

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

**Prompting:** Use *"a man"* or *"a woman"* to avoid altering facial features. Prompts support both English and Chinese. Use for frontal, medium-to-close-up videos.
**Input Image:** Use a high-resolution frontal face image for the best results. Our built-in pipeline handles different resolutions/formats.

### Inference with Community LoRA

Load LoRA models alongside Stand-In using `infer_with_lora.py`:

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

Recommended stylization LoRA:  [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

### Video Face Swapping

Perform face swapping with the experimental `infer_face_swap.py`:

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note:** This feature is experimental.
Adjust `--denoising_strength` for background and face blending.  Use `--force_background_consistency` cautiously, as it may cause contour issues.

### Infer with VACE

Generate videos with VACE and Stand-In via `infer_with_vace.py`:

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

VACE's control has a preset bias towards faces, which affects identity preservation. Please lower the `vace_scale` to a balance point where both motion and identity are preserved. When only `ip_image` and `reference_video` are provided, the weight can be reduced to 0.5.

Using both Stand-In and VACE together is more challenging than using Stand-In alone. We are still maintaining this feature, so if you encounter unexpected outputs or have other questions, feel free to raise them in the issue.

---

## Acknowledgements

This project is built upon the following excellent open-source projects:
*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We sincerely thank the authors and contributors of these projects.

The original raw material of our dataset was collected with the help of our team member [Binxin Yang](https://binxinyang.github.io/), and we appreciate his contribution!

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

For any questions or suggestions, open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues).

**Visit the original repository for more information: [https://github.com/WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In)**