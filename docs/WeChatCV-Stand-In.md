<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly preserve identity in your videos with Stand-In, a plug-and-play solution that requires minimal training.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)
[![GitHub](https://img.shields.io/github/stars/WeChatCV/Stand-In?style=social)](https://github.com/WeChatCV/Stand-In)

</div>

[**Explore the Stand-In repository on GitHub for more details.**](https://github.com/WeChatCV/Stand-In)

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

**Stand-In** is a groundbreaking framework for identity-preserving video generation, offering state-of-the-art results while requiring only **1%** additional training parameters compared to the base video generation model. This innovative approach achieves superior Face Similarity and Naturalness, surpassing full-parameter training methods.  Seamlessly integrate Stand-In into various applications, including:

*   Subject-Driven Video Generation
*   Pose-Controlled Video Generation
*   Video Stylization
*   Face Swapping

---

## 🚀 Key Features

*   **Lightweight & Efficient:** Train only a small fraction (1%) of the base model's parameters.
*   **Exceptional Fidelity:** Achieve outstanding identity consistency without compromising video quality.
*   **Plug-and-Play Integration:** Easily integrates with existing Text-to-Video (T2V) models.
*   **Highly Extensible:** Compatible with community models like LoRA and supports various downstream video tasks.

---

## 📰 What's New

*   **[2025.08.18]** Released VACE compatibility.  Now you can combine pose control and depth maps with Stand-In to preserve identity.
*   **[2025.08.16]** Updated experimental face swapping feature.
*   **[2025.08.13]** Official Stand-In preprocessor ComfyUI node released:  [https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI)
*   **[2025.08.12]** Released Stand-In v1.0 (153M parameters) with open-sourced Wan2.1-14B-T2V-adapted weights and inference code.

---

## ✨ Showcase: Example Outputs

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt                                                                                                                                                                                                                                                                    | Generated Video                                                                                                                                                         |
| :-------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  ![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)  | "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |  ![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3)  |
|  ![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)  | "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." |  ![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab)  |

---

### Non-Human Subject-Preserving Video Generation

| Reference Image                                                                                                  | Prompt                                                                                                                            | Generated Video                                                                                                   |
| :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------: |
|  <img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" />  | "A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads." |  ![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5)  |

---

### Identity-Preserving Stylized Video Generation

| Reference Image                                                                | LoRA        | Generated Video                                                                         |
| :-----------------------------------------------------------------------------: | :----------: | :---------------------------------------------------------------------------------------: |
|  ![Image](https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1)  | Ghibli LoRA |  ![Image](https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f)  |

---

### Video Face Swapping

| Reference Video                                                                   | Identity                                                                                    | Generated Video                                                                                                        |
| :--------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
|  ![Image](https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701)  |  <img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" />  |  ![Image](https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50)  |

---

### Pose-Guided Video Generation (With VACE)

| Reference Pose                                                                                                  | First Frame                                                                                                  | Generated Video                                                                                     |
| :---------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
|  ![Image](https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94)  |  <img width="719" height="415" alt="Image" src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" />  |  ![Image](https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44)  |

---

### Discover More
For more examples and results, please visit the project's website:  [https://www.Stand-In.tech](https://www.Stand-In.tech)

---

## 🛠️ Getting Started

### 1.  Environment Setup

```bash
# Clone the repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate a Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install the necessary dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for accelerated inference
# Note:  Ensure compatibility between your GPU and CUDA version with Flash Attention
pip install flash-attn --no-build-isolation
```

### 2.  Model Download

Download all required model weights using the automated download script:

```bash
python download_models.py
```

This script will automatically download:

*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (Stand-In model)

>  **Important:** If you already have the `wan2.1-T2V-14B` model, you can manually edit the `download_model.py` script to prevent re-downloading it. Place your existing model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## 💻 Usage Instructions

### Standard Inference
Generate identity-preserving videos from text prompts using the `infer.py` script:

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```
**Prompting Tips:**  For unchanged facial features, use a simple prompt like *"a man"* or *"a woman"*.  Prompts support both English and Chinese. The output will generate a medium to close-up frontal view video.

**Input Image Recommendation:**  Use a high-resolution frontal face image for optimal results.  The built-in pre-processing pipeline automatically handles image resolution and file formats.

### Inference with Community LoRA
Use the `infer_with_lora.py` script to combine Stand-In with LoRA models:

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```
**Recommended LoRA:**  Explore the Studio Ghibli LoRA at  [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b).

### Video Face Swapping
Experiment with video face swapping using the `infer_face_swap.py` script:

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```

**Note**: Face swapping is still experimental.  Adjust `--denoising_strength` for optimal results. Higher values redraw more of the background and make the face appear more natural. Lower values preserve the background but may result in overfitting of the face. Use `--force_background_consistency` with caution, as it may cause visible contour issues.

### Inference with VACE

Incorporate VACE-based controls for pose-guided video generation using the `infer_with_vace.py` script:

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
**Dependencies**: Download the required weights from the [VACE repository](link to VACE repo) or specify the path to the VACE weights using the `vace_path` parameter.

```bash
python download_models.py --vace
```
**Input Preprocessing**:  Preprocess your input control video using the VACE preprocessing tool.  Both `reference_video` and `reference_image` are optional.  Experiment with the `vace_scale` to balance motion and identity preservation.  When using just the `ip_image` and `reference_video`, consider reducing the scale to 0.5.

*   **Important:** Integrating Stand-In with VACE can be complex.  Please report any unexpected outputs or questions via the issue tracker.

---

## 🙏 Acknowledgements

This project builds upon the foundations of these amazing open-source projects:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We extend our sincere gratitude to the authors and contributors of these valuable projects.

Special thanks to [Binxin Yang](https://binxinyang.github.io/) for providing the original dataset materials!

---

## ✍️ Citation

If you utilize Stand-In in your research, kindly cite our paper:

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

For any questions or suggestions, don't hesitate to open an issue on [GitHub](https://github.com/WeChatCV/Stand-In/issues). Your feedback is highly valued!