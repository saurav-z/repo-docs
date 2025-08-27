# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text or Images

**Create stunning, explorable 3D worlds from text or images with HunyuanWorld 1.0, a cutting-edge, open-source model from Tencent.**  [View the original repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage&height=22px)](https://3d.hunyuan.tencent.com/sceneTo3D)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg?height=22px)](https://huggingface.co/tencent/HunyuanWorld-1)
[![Page](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github&height=22px)](https://3d-models.hunyuan.tencent.com/world/)
[![Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv&height=22px)](https://arxiv.org/abs/2507.21809)
[![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord&height=22px)](https://discord.gg/dNBrdrGGMa)
[![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x&height=22px)](https://x.com/TencentHunyuan)
[![Community](https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore&height=22px)](#community-resources)

<br>

<p align="center">
  "To see a World in a Grain of Sand, and a Heaven in a Wild Flower"
</p>

## Key Features

*   **3D World Generation:** Generate interactive and explorable 3D worlds from text prompts or images.
*   **Immersive 360° Experiences:** Create immersive experiences using panoramic world proxies.
*   **Mesh Export:** Generate meshes for seamless integration with existing computer graphics pipelines.
*   **Disentangled Object Representations:** Enhanced interactivity through disentangled object representations.
*   **State-of-the-Art Performance:**  Achieves superior results in generating coherent, explorable, and interactive 3D worlds.
*   **Quantization Support:** Includes a lite (quantized) version (HunyuanWorld-1.0-lite) to run on consumer-grade GPUs like the 4090.

## 🔥 What's New

*   **August 15, 2025:** Released the quantized version of HunyuanWorld-1.0 (HunyuanWorld-1.0-lite), optimized for consumer GPUs (e.g., 4090)!
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809).
*   **July 26, 2025:** Launched HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model.

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> | 

## ☯️ **HunyuanWorld 1.0 Overview**

### Abstract

HunyuanWorld 1.0 addresses the challenge of generating interactive 3D worlds from text or images, combining the strengths of video-based and 3D-based approaches. This innovative framework enables the creation of immersive, explorable, and interactive 3D environments with three key advantages:  360° panoramic experiences, mesh export capabilities, and disentangled object representations.  It utilizes a semantically layered 3D mesh representation and panoramic images as 360° world proxies for advanced scene decomposition and reconstruction, achieving state-of-the-art results in generating diverse and engaging 3D worlds suitable for virtual reality, game development, and content creation.

<p align="center">
  <img src="assets/application.png">
</p>

### Architecture

The generation architecture of Tencent HunyuanWorld-1.0 integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction. This enables the generation of high-quality, scene-scale 360° 3D worlds, supporting both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg">
</p>

### Performance

HunyuanWorld 1.0 consistently outperforms other open-source panorama generation and 3D world generation methods.

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

Experience the immersive and explorable 3D worlds generated by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif">
</p>

 <p align="left">
  <img src="assets/panorama2.gif">
</p> 

<p align="left">
  <img src="assets/roaming_world.gif">
</p>

## 🎁 Model Zoo

HunyuanWorld 1.0 is based on Flux, and can be adapted to other image generation models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Getting Started

Follow these steps to use HunyuanWorld 1.0:

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

**Image to World Generation:**

```python
# 1. Generate a Panorama image from an Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# 2. Create a World Scene using the Panorama image with HunyuanWorld 1.0.
#    Specify foreground object labels using --labels_fg1 & --labels_fg2.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
#  View your WORLD SCENE!
```

**Text to World Generation:**

```python
# 1. Generate a Panorama image from a Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# 2. Create a World Scene using the Panorama image with HunyuanWorld 1.0.
#    Specify foreground object labels using --labels_fg1 & --labels_fg2.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# View your WORLD SCENE!
```

### Quantization & Cache Usage

**Image to World Generation (with Quantization/Cache):**

```python
# Step 1: Generate panorama with quantization and/or cache.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2: Generate world scene with quantization and/or cache.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

**Text to World Generation (with Quantization/Cache):**

```python
# Step 1: Generate panorama with quantization and/or cache.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
# Step 2: Generate world scene with quantization and/or cache.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start

Run a quick example:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Use the provided ```modelviewer.html``` to visualize your 3D worlds in a web browser.  Just upload the scene files and enjoy!

<p align="left">
  <img src="assets/quick_look.gif">
</p>

*Note: Scene loading may fail due to hardware limitations.*

## 📑 Open-Source Roadmap

*   \[x] Inference Code
*   \[x] Model Checkpoints
*   \[x] Technical Report
*   \[x] Lite Version
*   \[ ] Voyager (RGBD Video Diffusion)

## 🔗 BibTeX

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

## 📧 Contact

For questions, email tengfeiwang12@gmail.com.

## 🙏 Acknowledgements

We thank the contributors to the [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen) repositories for their open research.