<!-- Improved README for HunyuanWorld-1.0 -->

# HunyuanWorld 1.0: Generate Immersive 3D Worlds with Text or Images

**Create explorable and interactive 3D worlds with text or images using the innovative HunyuanWorld 1.0 model!** [See the original repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld-1.0 Teaser">
</p>

<div align="center">
  <a href="https://3d.hunyuan.tencent.com/sceneTo3D" target="_blank"><img src="https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage" height="22px" alt="Official Site"></a>
  <a href="https://huggingface.co/tencent/HunyuanWorld-1" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg" height="22px" alt="Hugging Face Models"></a>
  <a href="https://3d-models.hunyuan.tencent.com/world/" target="_blank"><img src="https://img.shields.io/badge/Page-bb8a2e.svg?logo=github" height="22px" alt="Model Page"></a>
  <a href="https://arxiv.org/abs/2507.21809" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="22px" alt="Research Report"></a>
  <a href="https://discord.gg/dNBrdrGGMa" target="_blank"><img src="https://img.shields.io/badge/Discord-white.svg?logo=discord" height="22px" alt="Discord"></a>
  <a href="https://x.com/TencentHunyuan" target="_blank"><img src="https://img.shields.io/badge/Hunyuan-black.svg?logo=x" height="22px" alt="X"></a>
 <a href="#community-resources" target="_blank"><img src="https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore" height="22px" alt="Community Resources"></a>
</div>

## Key Features

*   **Immersive 3D World Generation:** Transform text and images into explorable 3D environments.
*   **Panoramic World Proxies:** Experience 360° immersive views for a rich, detailed world.
*   **Mesh Export Capabilities:** Seamless integration with existing computer graphics pipelines.
*   **Disentangled Object Representations:** Enhanced interactivity and customization of generated scenes.
*   **State-of-the-Art Performance:** Achieve superior visual quality and geometric consistency in world generation.

## 🔥 News

*   **July 26, 2025:** HunyuanWorld-1.0's [technical report](https://arxiv.org/abs/2507.21809) released, with details and insights into the groundbreaking technology.
*   **July 26, 2025:** Introducing HunyuanWorld-1.0: the first open-source, simulation-capable, and immersive 3D world generation model!

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** community to discuss and get support.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140 alt="Wechat QR Code"> | <img src="assets/qrcode/xiaohongshu.png"  height=140 alt="Xiaohongshu QR Code"> | <img src="assets/qrcode/x.png"  height=140 alt="X QR Code"> | <img src="assets/qrcode/discord.png"  height=140 alt="Discord QR Code"> |

## ☯️ **HunyuanWorld 1.0 Overview**

### Abstract

HunyuanWorld 1.0 addresses the challenge of generating playable 3D worlds from text or images. This framework combines the strengths of video-based (diversity) and 3D-based (consistency) methods. It offers:

*   **360° immersive experiences** via panoramic world proxies.
*   **Mesh export capabilities** for compatibility with existing graphics pipelines.
*   **Disentangled object representations** for augmented interactivity.

The core is a semantically layered 3D mesh representation that leverages panoramic images as 360° world proxies for semantic-aware world decomposition and reconstruction. Extensive experiments show state-of-the-art performance in generating coherent, explorable, and interactive 3D worlds.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld-1.0 Application Examples">
</p>

### Architecture

The architecture of HunyuanWorld-1.0 integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction for high-quality scene-scale 360° 3D world generation, supporting both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg" alt="HunyuanWorld-1.0 Architecture">
</p>

### Performance

HunyuanWorld 1.0 outperforms other open-source panorama and 3D world generation methods in visual quality and geometric consistency.

**[Detailed performance tables for Text-to-Panorama, Image-to-Panorama, Text-to-World, and Image-to-World generation are included in the original README]**

### Visual Results

Experience the generated 360° immersive and explorable 3D worlds created by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama 1 Example">
</p>

 <p align="left">
  <img src="assets/panorama2.gif" alt="Panorama 2 Example">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World Example">
</p>

## 🎁 Models Zoo

HunyuanWorld 1.0 open-source version is based on Flux and can be adapted to other image generation models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started

Follow these steps to get started with HunyuanWorld 1.0:

### Environment Setup

Tested with Python 3.10 and PyTorch 2.5.0+cu124.

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml

# real-esrgan install
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr-fixed
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

# zim anything install & download ckpt from ZIM project page
cd ..
git clone https://github.com/naver-ai/ZIM.git
cd ZIM; pip install -e .
mkdir zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx

# TO export draco format, you should install draco first
cd ../..
git clone https://github.com/google/draco.git
cd draco
mkdir build
cd build
cmake ..
make
sudo make install

# login your own hugging face account
cd ../..
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### Code Usage

*   **Image-to-World Generation:**

```python
# Generate Panorama image from an Image
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene with HunyuanWorld 1.0 using the Panorama
# Indicate foreground object labels using params labels_fg1 & labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2
# View the Generated World Scene
```

*   **Text-to-World Generation:**

```python
# Generate a Panorama image from a Prompt
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene with HunyuanWorld 1.0 using the Panorama
# Indicate foreground object labels using params labels_fg1 & labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# View the Generated World Scene
```

### Quick Start

Run the provided example for a quick start:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Use the ModelViewer tool to visualize your generated 3D worlds in a web browser.

1.  Open ```modelviewer.html``` in your browser.
2.  Upload the generated 3D scene files.
3.  Enjoy the real-time experience.

<p align="left">
  <img src="assets/quick_look.gif" alt="3D World Viewer">
</p>

*Note: Certain scenes might fail to load due to hardware limitations.*

## 📑 Open-Source Plan

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [ ] TensorRT Version
*   [ ] RGBD Video Diffusion

## 🔗 BibTeX

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

## Acknowledgements

We extend our gratitude to the contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their open research contributions.