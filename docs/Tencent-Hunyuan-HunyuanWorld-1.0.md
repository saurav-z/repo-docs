# HunyuanWorld 1.0: Generate Interactive 3D Worlds from Text or Images

**Experience a new era of immersive 3D world creation with HunyuanWorld 1.0, the first open-source model capable of generating explorable and interactive 3D worlds from text and images.**  Dive into a world of possibilities and bring your imagination to life!  [Explore the original repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld 1.0 Teaser">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px alt="Official Site"></a>
  <a href=https://huggingface.co/tencent/HunyuanWorld-1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px alt="Models on Hugging Face"></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px alt="3D Models Page"></a>
  <a href=https://arxiv.org/abs/2507.21809 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px alt="Technical Report"></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px alt="Discord"></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px alt="Hunyuan on X"></a>
  <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px alt="Community Resources"></a>
</div>

## Key Features

*   **Text-to-3D and Image-to-3D World Generation:** Create immersive 3D environments from textual descriptions or existing images.
*   **360° Panoramic World Proxies:** Experience fully immersive worlds with 360-degree coverage.
*   **Mesh Export Capabilities:** Seamlessly integrate generated worlds into existing computer graphics pipelines.
*   **Disentangled Object Representations:** Enable advanced interactivity within your 3D worlds.
*   **Quantization Support:**  Run the model on consumer-grade GPUs (like the 4090) using the -lite version.
*   **Caching Support:**  Speed up inference times.

## 🔥 What's New

*   **August 15, 2025:**  HunyuanWorld-1.0-lite, the quantization version, has been released, supporting consumer-grade GPUs.
*   **July 26, 2025:** The [technical report](https://arxiv.org/abs/2507.21809) for HunyuanWorld-1.0 was released.
*   **July 26, 2025:** HunyuanWorld-1.0, a groundbreaking, open-source, simulation-capable 3D world generation model, was launched.

> Connect with the community on **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** for support and discussions.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140 alt="Wechat QR Code"> | <img src="assets/qrcode/xiaohongshu.png"  height=140 alt="Xiaohongshu QR Code"> | <img src="assets/qrcode/x.png"  height=140 alt="X QR Code"> | <img src="assets/qrcode/discord.png"  height=140 alt="Discord QR Code"> |

## ☯️ **HunyuanWorld 1.0: Technical Overview**

### Abstract

HunyuanWorld 1.0 tackles the challenge of generating realistic and interactive 3D worlds from text and images. It combines the strengths of video-based and 3D-based methods, offering both rich diversity and geometric consistency.  The framework utilizes a semantically layered 3D mesh representation and panoramic images to create immersive 360° experiences, enable mesh export, and support disentangled object representations for enhanced interactivity. This results in state-of-the-art performance across various applications, including VR, simulation, and game development.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld 1.0 Applications">
</p>

### Architecture

HunyuanWorld-1.0's architecture leverages panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction for high-quality 360° scene generation, supporting both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg" alt="HunyuanWorld 1.0 Architecture Diagram">
</p>

### Performance

HunyuanWorld 1.0 achieves superior results compared to other panorama and 3D world generation methods across multiple metrics, including BRISQUE, NIQE, Q-Align, and CLIP scores.

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

### Visual Results

Experience the generated 360° immersive 3D worlds:

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama 1 Example">
</p>

 <p align="left">
  <img src="assets/panorama2.gif" alt="Panorama 2 Example">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World Example">
</p>

## 🎁 Model Zoo

HunyuanWorld 1.0 is based on Flux and can be adapted to image generation models such as Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Getting Started

### Environment Setup

Tested with Python 3.10 and PyTorch 2.5.0+cu124.

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml

# Real-ESRGAN Install (For Image Enhancement)
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr-fixed
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

# ZIM (For Depth Estimation - if used)
cd ..
git clone https://github.com/naver-ai/ZIM.git
cd ZIM; pip install -e .
mkdir zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx

# Draco Install (For Mesh Export)
cd ../..
git clone https://github.com/google/draco.git
cd draco
mkdir build
cd build
cmake ..
make
sudo make install

# Hugging Face Login (Required to download models)
cd ../..
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### Code Examples

**Image-to-World Generation:**

```python
# Generate Panorama
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Generate World Scene
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
#  Result: Your WORLD SCENE!
```

**Text-to-World Generation:**

```python
# Generate Panorama
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Generate World Scene
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case7
#  Result: Your WORLD SCENE!
```

### Quantization and Caching

**Quantization & Cache** for optimized memory usage and speed.

```python
# Quantization Example
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention

# Cache Example
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache

# Generate Scene using quantization/cache (Adapt the input path from the step above)
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention

CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

### Quick Start

Get started quickly with:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Visualize your generated 3D worlds in a web browser:

1.  Open `modelviewer.html` in your browser.
2.  Upload your generated 3D scene files.
3.  Enjoy the real-time play experiences!

<p align="left">
  <img src="assets/quick_look.gif" alt="3D World Viewer">
</p>

*Note: Certain scenes may fail to load due to hardware limitations.*

## 📑 Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [x] Lite Version
*   [ ] Voyager (RGBD Video Diffusion)

## 🔗 Citation

```
@misc{hunyuanworld2025tencent,
    title={HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels},
    author={Tencent, HunyuanWorld Team},
    year={2025},
    eprint={2507.21809},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

For any questions, please contact tengfeiwang12@gmail.com.

## Acknowledgements

The developers would like to thank the contributors to the following repositories: [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen).