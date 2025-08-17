# HunyuanWorld 1.0: Create Immersive 3D Worlds from Text or Images

**Generate stunning, explorable 3D worlds from simple text descriptions or images with Tencent's HunyuanWorld 1.0!** [Explore the original repo](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld Teaser">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px alt="Official Site"></a>
  <a href=https://huggingface.co/tencent/HunyuanWorld-1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px alt="Hugging Face Models"></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px alt="3D Models Page"></a>
  <a href=https://arxiv.org/abs/2507.21809 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px alt="Research Report"></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px alt="Discord Community"></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px alt="Hunyuan Twitter"></a>
  <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px alt="Community Resources"></a>
</div>

<br>

> "To see a World in a Grain of Sand, and a Heaven in a Wild Flower"

<br>

<img src="https://github.com/user-attachments/assets/blob/main/assets/747b3e41-df9c-4cd2-b1d1-c0dce63f63ef.png" alt="HunyuanWorld Example" width="100%">

## Key Features

*   **Text-to-3D & Image-to-3D Generation:** Transform text prompts or images into interactive 3D worlds.
*   **Immersive 360° Experiences:** Explore generated worlds with panoramic world proxies.
*   **Mesh Export Capabilities:** Seamlessly integrate with existing computer graphics pipelines.
*   **Disentangled Object Representations:** Enables augmented interactivity within the generated scenes.
*   **State-of-the-Art Performance:** Outperforms existing methods in visual quality and geometric consistency.
*   **Quantization & Cache Support:** Optimized for memory usage and faster inference.

## What's New

*   **August 15, 2025:** Released HunyuanWorld-1.0-lite, a quantization version, now compatible with consumer-grade GPUs like the 4090!
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809) detailing the advancements of HunyuanWorld-1.0.
*   **July 26, 2025:** Launched HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model.

> Join the **[Discord](https://discord.gg/dNBrdrGGMa)** to discuss, ask questions, and receive assistance.

<table align="center">
  <tr>
    <td>
        <img src="assets/qrcode/wechat.png"  height=140 alt="WeChat QR Code">
    </td>
    <td>
        <img src="assets/qrcode/xiaohongshu.png"  height=140 alt="Xiaohongshu QR Code">
    </td>
    <td>
        <img src="assets/qrcode/x.png"  height=140 alt="X QR Code">
    </td>
    <td>
        <img src="assets/qrcode/discord.png"  height=140 alt="Discord QR Code">
    </td>
  </tr>
</table>

## ☯️ **HunyuanWorld 1.0 Overview**

### Abstract

HunyuanWorld 1.0 is a groundbreaking framework for generating explorable and interactive 3D worlds. The framework leverages panoramic images as 360° world proxies for semantic-aware world decomposition and reconstruction.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld Applications">
</p>

### Architecture

The architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction for high-quality scene-scale 360° 3D world generation, supporting both text and image inputs.

<p align="center">
  <img src="assets/arch.jpg" alt="HunyuanWorld Architecture">
</p>

### Performance

HunyuanWorld 1.0 showcases superior performance compared to other state-of-the-art open-source methods in panorama and 3D world generation.

#### Text-to-Panorama Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| Diffusion360     | 69.5       | 7.5      | 1.8        | 20.9      |
| MVDiffusion      | 47.9       | 7.1      | 2.4        | 21.5      |
| PanFusion        | 56.6       | 7.6      | 2.2        | 21.0      |
| LayerPano3D      | 49.6       | 6.5      | 3.7        | 21.5      |
| HunyuanWorld 1.0 | **40.8**   | **5.8**  | **4.4**    | **24.3**  |

#### Image-to-Panorama Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| Diffusion360     | 71.4       | 7.8      | 1.9        | 73.9      |
| MVDiffusion      | 47.7       | 7.0      | 2.7        | 80.8      |
| HunyuanWorld 1.0 | **45.2**   | **5.8**  | **4.3**    | **85.1**  |

#### Text-to-World Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| Director3D       | 49.8       | 7.5      | 3.2        | 23.5      |
| LayerPano3D      | 35.3       | 4.8      | 3.9        | 22.0      |
| HunyuanWorld 1.0 | **34.6**   | **4.3**  | **4.2**    | **24.0**  |

#### Image-to-World Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| WonderJourney    | 51.8       | 7.3      | 3.2        | 81.5      |
| DimensionX       | 45.2       | 6.3      | 3.5        | 83.3      |
| HunyuanWorld 1.0 | **36.2**   | **4.6**  | **3.9**    | **84.5**  |

### Visual Results

Experience the quality of generated 3D worlds through interactive GIFs:

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama 1">
</p>

<p align="left">
  <img src="assets/panorama2.gif" alt="Panorama 2">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World">
</p>

## 🎁 Models Zoo

Explore the available models to start generating 3D worlds. The open-source version of HY World 1.0 is based on Flux and can be adapted to image generation models such as Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                  | Date       | Size   | Hugging Face                                                                                                 |
| ------------------------------ | ------------------------------ | ---------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model       | 2025-07-26 | 478MB  | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)               |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model      | 2025-07-26 | 478MB  | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)              |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene  | 2025-07-26 | 478MB  | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene)          |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky    | 2025-07-26 | 120MB  | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)            |

## 🤗 Get Started with HunyuanWorld 1.0

### Environment Setup

Ensure you have Python 3.10 and PyTorch 2.5.0+cu124:

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

#### Image-to-World Generation:

```python
# Generate a Panorama image with An Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2
# And then you get your WORLD SCENE!!
```

#### Text-to-World Generation:

```python
# Generate a Panorama image with A Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# And then you get your WORLD SCENE!!
```

#### Quantization & Cache Usage:

```python
# For Image-to-World:
# Step 1: Use Quantization or Cache options
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention  # Quantization
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache                                 # Cache
# Step 2: Use Quantization or Cache options
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention # Quantization
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache # Cache

# For Text-to-World:
# Step 1: Use Quantization or Cache options
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention # Quantization
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache                               # Cache
# Step 2: Use Quantization or Cache options
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention # Quantization
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache # Cache
```

### Quick Start

Get started fast with example scripts:

```bash
bash scripts/test.sh
```

### 3D World Viewer

View your generated 3D worlds directly in your browser using the provided ModelViewer tool. Open ```modelviewer.html``` and upload your 3D scene files for real-time exploration.

<p align="left">
  <img src="assets/quick_look.gif" alt="3D World Viewer">
</p>

Note: Some scenes may not load due to hardware constraints.

## 📑 Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [ ] TensorRT Version
*   [ ] RGBD Video Diffusion

## 🔗 BibTeX

```
@misc{hunyuanworld2025tencent,
    title={HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels},
    author={Tencent HunyuanWorld Team},
    year={2025},
    eprint={2507.21809},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

For questions, please email tengfeiwang12@gmail.com.

## Acknowledgements

Special thanks to the contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their valuable contributions to open research.