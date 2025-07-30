# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text and Images

**Create breathtaking 3D worlds from your imagination with HunyuanWorld 1.0, the revolutionary AI model from Tencent!** ([Original Repo](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0))

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld 1.0 Teaser">
</p>

<div align="center">
  <a href="https://3d.hunyuan.tencent.com/sceneTo3D" target="_blank"><img src="https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage" height="22px" alt="Official Site"></a>
  <a href="https://huggingface.co/tencent/HunyuanWorld-1" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg" height="22px" alt="Hugging Face Models"></a>
  <a href="https://3d-models.hunyuan.tencent.com/world/" target="_blank"><img src="https://img.shields.io/badge/Page-bb8a2e.svg?logo=github" height="22px" alt="Model Page"></a>
  <a href="https://arxiv.org/abs/2507.21809" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="22px" alt="Technical Report"></a>
  <a href="https://discord.gg/dNBrdrGGMa" target="_blank"><img src="https://img.shields.io/badge/Discord-white.svg?logo=discord" height="22px" alt="Discord"></a>
  <a href="https://x.com/TencentHunyuan" target="_blank"><img src="https://img.shields.io/badge/Hunyuan-black.svg?logo=x" height="22px" alt="X"></a>
 <a href="#community-resources" target="_blank"><img src="https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore" height="22px" alt="Community Resources"></a>
</div>

<br>

> "To see a World in a Grain of Sand, and a Heaven in a Wild Flower"

<img src="https://github.com/user-attachments/assets/747b3e41-df9c-4cd2-b1d1-c0dce63f63ef" alt="Quote Image" width="600">

## ✨ Key Features

*   **Immersive 360° Worlds:** Experience fully explorable environments.
*   **Mesh Export:** Compatible with existing computer graphics pipelines.
*   **Interactive Objects:** Disentangled object representations for dynamic interaction.
*   **Text-to-World & Image-to-World:** Generate worlds from text prompts or images.
*   **State-of-the-Art Performance:** Outperforms existing methods in visual quality and consistency.
*   **Open-Source:** Access models, code, and a technical report to get started.

## 📰 What's New

*   **July 26, 2025:** Technical Report released, delving into the details of HunyuanWorld-1.0.  [Technical Report](https://3d-models.hunyuan.tencent.com/world/HY_World_1_technical_report.pdf)
*   **July 26, 2025:** HunyuanWorld-1.0 is now open-source!

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** to discuss and get help.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> |

## ☯️ HunyuanWorld 1.0: Overview

### Abstract

HunyuanWorld 1.0 addresses the challenge of generating immersive, playable 3D worlds from text or images. The model combines the strengths of video-based and 3D-based methods, providing both rich diversity and geometric consistency. Key features include 360° immersive experiences, mesh export capabilities, and disentangled object representations for interactivity. Utilizing a semantically layered 3D mesh representation, the model generates diverse 3D worlds.

<p align="center">
  <img src="assets/application.png" alt="Application Example">
</p>

### Architecture

The generation architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to create high-quality scene-scale 360° 3D worlds from text and image inputs.

<p align="left">
  <img src="assets/arch.jpg" alt="Architecture Diagram">
</p>

### Performance

HunyuanWorld 1.0 showcases superior performance compared to other open-source panorama and 3D world generation methods.

**Text-to-panorama generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 69.5                  | 7.5                | 1.8                 | 20.9               |
| MVDiffusion      | 47.9                  | 7.1                | 2.4                 | 21.5               |
| PanFusion        | 56.6                  | 7.6                | 2.2                 | 21.0               |
| LayerPano3D      | 49.6                  | 6.5                | 3.7                 | 21.5               |
| HunyuanWorld 1.0 | **40.8**              | **5.8**            | **4.4**             | **24.3**           |

**Image-to-panorama generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 71.4                  | 7.8                | 1.9                 | 73.9               |
| MVDiffusion      | 47.7                  | 7.0                | 2.7                 | 80.8               |
| HunyuanWorld 1.0 | **45.2**              | **5.8**            | **4.3**             | **85.1**           |

**Text-to-world generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Director3D       | 49.8                  | 7.5                | 3.2                 | 23.5               |
| LayerPano3D      | 35.3                  | 4.8                | 3.9                 | 22.0               |
| HunyuanWorld 1.0 | **34.6**              | **4.3**            | **4.2**             | **24.0**           |

**Image-to-world generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| WonderJourney    | 51.8                  | 7.3                | 3.2                 | 81.5               |
| DimensionX       | 45.2                  | 6.3                | 3.5                 | 83.3               |
| HunyuanWorld 1.0 | **36.2**              | **4.6**            | **3.9**             | **84.5**           |

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama Example 1">
</p>

<p align="left">
  <img src="assets/panorama2.gif" alt="Panorama Example 2">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World Example">
</p>

## 📦 Model Zoo

Find pre-trained models to get started quickly.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🚀 Getting Started

Follow these steps to use HunyuanWorld 1.0:

### 1. Environment Setup

Ensure you have Python 3.10 and PyTorch 2.5.0+cu124 installed.

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml

# Install Real-ESRGAN
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr-fixed
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

# Install ZIM
cd ..
git clone https://github.com/naver-ai/ZIM.git
cd ZIM; pip install -e .
mkdir zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx

# Install Draco (for mesh export)
cd ../..
git clone https://github.com/google/draco.git
cd draco
mkdir build
cd build
cmake ..
make
sudo make install

# Login to Hugging Face
cd ../..
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### 2. Code Usage

**Image to World Generation:**

```python
# Generate a Panorama image from an Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene using the Panorama image
# Specify foreground object labels with --labels_fg1 & --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# View the generated WORLD SCENE!
```

**Text to World Generation:**

```python
# Generate a Panorama image from a Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene using the Panorama image
# Specify foreground object labels with --labels_fg1 & --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# View the generated WORLD SCENE!
```

### 3. Quick Start

Run the following command for a quick example:

```python
bash scripts/test.sh
```

### 4. 3D World Viewer

Visualize your creations in your web browser. Open ```modelviewer.html```, upload the generated 3D scene files, and enjoy!

<p align="left">
  <img src="assets/quick_look.gif" alt="Quick Look at the 3D World Viewer">
</p>

*Note: Some scenes may fail to load due to hardware limitations.*

## 🚧 Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [ ] TensorRT Version (Coming Soon)
*   [ ] RGBD Video Diffusion (Planned)

## 📚 Citation

```bibtex
@misc{hunyuanworld2025tencent,
    title={HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels},
    author={Tencent HunyuanWorld Team},
    year={2025},
    eprint={2507.21809},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## 🙏 Acknowledgements

We appreciate the contributions of the following open-source projects: [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen).