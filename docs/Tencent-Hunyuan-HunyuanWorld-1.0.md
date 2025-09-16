# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text and Images

**Create stunning, explorable 3D worlds from text and images with HunyuanWorld 1.0, revolutionizing virtual reality, game development, and interactive content creation.**

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px)](https://3d.hunyuan.tencent.com/sceneTo3D)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px)](https://huggingface.co/tencent/HunyuanWorld-1)
[![3D Model Page](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px)](https://3d-models.hunyuan.tencent.com/world/)
[![Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px)](https://arxiv.org/abs/2507.21809)
[![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px)](https://discord.gg/dNBrdrGGMa)
[![X (Twitter)](https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px)](https://x.com/TencentHunyuan)
[![Community Resources](https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px)](#community-resources)

[View the original repository on GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld Teaser Image">
</p>

## Key Features

*   **Immersive 360° Experiences:** Generate panoramic world proxies for truly immersive explorations.
*   **Mesh Export Capabilities:** Seamlessly integrate with existing computer graphics pipelines.
*   **Disentangled Object Representations:** Enables enhanced interactivity within generated worlds.
*   **Text-to-World & Image-to-World Generation:** Create 3D worlds from text prompts or images.
*   **Open Source and Accessible:**  Includes inference code, model checkpoints, and a "lite" version for broader hardware compatibility.

## 🔥 What's New

*   **September 2, 2025:** Released RGB-D Video Diffusion model [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager/) for 3D-consistent world exploration and fast 3D reconstruction!
*   **August 15, 2025:** Released a quantized version (HunyuanWorld-1.0-lite) for consumer-grade GPUs (e.g., 4090).
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809).
*   **July 26, 2025:**  Launched HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model.

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> | 

## ☯️ **HunyuanWorld 1.0 Overview**

### Abstract

HunyuanWorld 1.0 addresses the challenge of generating playable 3D worlds by combining the strengths of video-based (rich diversity) and 3D-based (geometric consistency) methods. This framework generates immersive, explorable, and interactive 3D worlds. It achieves this through: panoramic world proxies for 360° experiences, mesh export capabilities for compatibility, and disentangled object representations for interactivity.  The core utilizes a semantically layered 3D mesh representation with panoramic images as 360° world proxies.  This approach enables diverse 3D world generation and sets a new standard for virtual reality, physical simulation, game development, and interactive content creation.

<p align="center">
  <img src="assets/application.png" alt="Application Example">
</p>

### Architecture

HunyuanWorld-1.0 utilizes an architecture that integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to generate high-quality, scene-scale 360° 3D worlds from both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg" alt="Architecture Diagram">
</p>

### Performance

HunyuanWorld 1.0 demonstrates state-of-the-art performance compared to other panorama and 3D world generation methods, as demonstrated by the benchmarks below:

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

Explore the impressive 360° immersive and explorable 3D worlds generated by HunyuanWorld 1.0:

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

The open-source version of HY World 1.0 is built on Flux, and easily adaptable to other image generation models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------| 
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started

### Environment Setup

Requires Python 3.10 and PyTorch 2.5.0+cu124.

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

**Image-to-World Generation:**

```python
# First, generate a Panorama image with An Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Second, using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2
# And then you get your WORLD SCENE!!
```

**Text-to-World Generation:**

```python
# First, generate a Panorama image with A Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Second, using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# And then you get your WORLD SCENE!!
```

### Quantization & Cache Usage

**Image-to-World Generation (with Quantization & Cache):**

```python
# Step 1:
# To optimize memory usage and speed up inference, quantization is a practical solution.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2:
# To optimize memory usage and speed up inference, quantization is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

**Text-to-World Generation (with Quantization & Cache):**

```python
# Step 1:
# To optimize memory usage and speed up inference, quantization is a practical solution.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
# Step 2:
# To optimize memory usage and speed up inference, quantization is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start

Get started quickly by running the example script:
```bash
bash scripts/test.sh
```

### 3D World Viewer

Visualize your generated 3D worlds in your web browser using the provided ModelViewer tool:

1.  Open `modelviewer.html` in your browser.
2.  Upload your generated 3D scene files.
3.  Enjoy real-time exploration!

<p align="left">
  <img src="assets/quick_look.gif" alt="Model Viewer Example">
</p>

*Note: Some scenes may fail to load due to hardware limitations.*

## 📑 Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [x] Lite Version
*   [x] Voyager (RGBD Video Diffusion)

## 🔗 BibTeX

```bibtex
@misc{hunyuanworld2025tencent,
    title={HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels},
    author={Tencent, HunyuanWorld Team},
    year={2025},
    eprint={2507.21809},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## 📧 Contact

For any questions, please contact tengfeiwang12@gmail.com.

## 🙏 Acknowledgements

We thank the contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their valuable open research.