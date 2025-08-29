# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text or Images

**Create explorable and interactive 3D worlds from text or images with HunyuanWorld 1.0, a cutting-edge open-source model from Tencent.**  ([Original Repo](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0))

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://3d.hunyuan.tencent.com/sceneTo3D)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg)](https://huggingface.co/tencent/HunyuanWorld-1)
[![Page](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github)](https://3d-models.hunyuan.tencent.com/world/)
[![Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2507.21809)
[![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord)](https://discord.gg/dNBrdrGGMa)
[![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x)](https://x.com/TencentHunyuan)
[![Community](https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore)](#community-resources)

<p align="center">
  <img src="assets/teaser.png">
</p>

<p align="center">
  "To see a World in a Grain of Sand, and a Heaven in a Wild Flower"
</p>

## Key Features

*   **Immersive 3D Worlds:** Generate 360° experiences with panoramic world proxies.
*   **Mesh Export:** Compatible with existing computer graphics pipelines for seamless integration.
*   **Interactive Objects:** Disentangled object representations for augmented interactivity.
*   **Text-to-World & Image-to-World:** Generate 3D worlds from both textual descriptions and images.
*   **Quantization Support:** Enhanced efficiency with a lite version supporting consumer-grade GPUs.

## What's New

*   **August 15, 2025:** Released the quantization version (HunyuanWorld-1.0-lite) for consumer-grade GPUs.
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809).
*   **July 26, 2025:** Launched the open-source HunyuanWorld-1.0, a simulation-capable 3D world generation model.

Join the community on [Discord](https://discord.gg/dNBrdrGGMa) for discussions and support.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> | 

## **HunyuanWorld 1.0 Overview**

### Abstract
HunyuanWorld 1.0 addresses the challenge of creating 3D worlds from text or images by combining the strengths of video-based and 3D-based methods. It offers immersive, explorable, and interactive 3D worlds, leveraging panoramic images as 360° world proxies. This approach provides 360° immersive experiences, mesh export capabilities, and disentangled object representations.

<p align="center">
  <img src="assets/application.png">
</p>

### Architecture

HunyuanWorld-1.0's generation architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to achieve high-quality scene-scale 360° 3D world generation, supporting both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg">
</p>

### Performance

HunyuanWorld 1.0 outperforms other methods in visual quality and geometric consistency, as demonstrated by benchmark results:

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | ---------- |
| HunyuanWorld 1.0 | **40.8**   | **5.8** | **4.4**    | **24.3**   |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | ---------- |
| HunyuanWorld 1.0 | **45.2**   | **5.8** | **4.3**    | **85.1**   |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | ---------- |
| HunyuanWorld 1.0 | **34.6**   | **4.3** | **4.2**    | **24.0**   |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | ---------- |
| HunyuanWorld 1.0 | **36.2**   | **4.6** | **3.9**    | **84.5**   |

### Visual Results

Explore the immersive 3D worlds created by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif">
</p>

 <p align="left">
  <img src="assets/panorama2.gif">
</p> 

<p align="left">
  <img src="assets/roaming_world.gif">
</p>

## Model Zoo

Access the pre-trained models for HunyuanWorld 1.0, based on Flux, and adaptable to models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------| 
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## Getting Started

Follow these steps to use HunyuanWorld 1.0:

### Environment Setup

Test environment: Python 3.10, PyTorch 2.5.0+cu124.

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
# Generate Panorama
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create World Scene
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2
```

**Text to World Generation:**

```python
# Generate Panorama
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create World Scene
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
```

### Quantization & Cache Usage

**Image to World Generation (with quantization/cache):**

```python
#Quantization
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
#Cache
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

**Text to World Generation (with quantization/cache):**

```python
#Quantization
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
#Cache
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start

Run the example script to get started quickly:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Visualize your generated 3D worlds in a web browser using the ModelViewer tool.  Open `modelviewer.html`, upload your scene files, and explore.

<p align="left">
  <img src="assets/quick_look.gif">
</p>

## Open-Source Plan

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [x] Lite Version
*   [ ] Voyager (RGBD Video Diffusion)

## BibTeX

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

## Community Resources

*   For any questions, please contact tengfeiwang12@gmail.com.

## Acknowledgements

Special thanks to the contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their valuable contributions.