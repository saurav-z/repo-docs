<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In: Lightweight Identity Control for Video Generation
  </h1>

  <h3>Effortlessly preserve identity in your videos with Stand-In, a plug-and-play solution that generates stunning results with minimal parameter adjustments.</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

Explore the power of Stand-In, a cutting-edge framework for identity-preserving video generation, offering state-of-the-art results with just 1% additional parameters compared to the base video generation model. This enables seamless integration into various video generation tasks, including subject-driven, pose-controlled, stylization, and face swapping, all while ensuring remarkable consistency. 

[Visit the original repository on GitHub](https://github.com/WeChatCV/Stand-In)

---

## Key Features

*   **Minimal Parameter Training:** Achieve impressive results with only a 1% increase in parameters.
*   **Superior Quality:** Maintain exceptional identity consistency without compromising video generation quality.
*   **Easy Integration:** Seamlessly integrate with existing text-to-video (T2V) models.
*   **Versatile Compatibility:** Works with community models (LoRA) and supports a wide range of video tasks.

---

## What's New

*   **[2025.08.18]**  VACE compatibility released, offering pose control and other control methods combined with Stand-In.
*   **[2025.08.16]**  Updated experimental version of the face swapping feature.
*   **[2025.08.13]**  Official Stand-In pre-processing ComfyUI node released: [Stand-In_Preprocessor_ComfyUI](https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI).
*   **[2025.08.12]** Stand-In v1.0 released, including the Wan2.1-14B-T2V adapted weights and inference code.

---

## Showcase

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

For more exciting results, visit our project page: [https://stand-in-video.github.io/](https://www.Stand-In.tech)

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

# (Optional) Install Flash Attention for faster inference
pip install flash-attn --no-build-isolation
```

### 2. Model Download

Run the automated download script to fetch all required model weights into the `checkpoints` directory.
```bash
python download_models.py
```
This downloads:
*   `wan2.1-T2V-14B` (base text-to-video model)
*   `antelopev2` (face recognition model)
*   `Stand-In` (our Stand-In model)

> Note: You can manually manage model downloads by editing `download_model.py` and placing the `wan2.1-T2V-14B` model in `checkpoints/wan2.1-T2V-14B` if you have it already.

---

## Usage

### Standard Inference

Use `infer.py` for identity-preserving text-to-video generation.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```
**Prompting Tip:** Use *"a man"* or *"a woman"* without extra descriptions to avoid altering facial features. Prompts are supported in both Chinese and English and are designed for frontal, medium-to-close-up videos.

**Input Image Recommendation:** Use a high-resolution frontal face image for optimal results. The built-in preprocessing pipeline handles various resolutions and file extensions.

---

### Inference with Community LoRA

Use `infer_with_lora.py` to combine Stand-In with community LoRA models.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

Recommended stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping

Use `infer_face_swap.py` for video face swapping with Stand-In.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```
**Note**:  Face swapping is experimental due to the lack of inpainting in Wan2.1.

Adjust `denoising_strength`:  Higher values redraw the background more, potentially leading to natural-looking faces. Lower values maintain the background but risk overfitting faces.  Consider `--force_background_consistency` carefully, as it might cause noticeable contour issues.  Experiment with `denoising_strength` for best results.

### Infer with VACE

Use `infer_with_vace.py` to perform identity-preserving video generation with Stand-In and VACE.
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
Download the weights from the VACE repository or provide the path to the VACE weights in the `vace_path` parameter.

Preprocess the input control video with VACE's tool. Both `reference_video` and `reference_image` are optional and can exist simultaneously. Reduce `vace_scale` to balance motion and identity preservation, especially because VACE's control has a preset bias towards faces. Lower the weight to 0.5 when only `ip_image` and `reference_video` are provided.

---

## Acknowledgements

This project is built upon the following excellent open-source projects:
*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We sincerely thank the authors and contributors of these projects.

Dataset material was collected with the help of our team member [Binxin Yang](https://binxinyang.github.io/), and we appreciate his contribution!

---

## Citation

If you find our work helpful for your research, please consider citing our paper:

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

For questions or suggestions, please open a [GitHub Issue](https://github.com/WeChatCV/Stand-In/issues). We value your feedback!