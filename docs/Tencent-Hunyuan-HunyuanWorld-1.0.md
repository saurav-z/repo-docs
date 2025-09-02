# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text or Images

**Create captivating and interactive 3D worlds with HunyuanWorld 1.0, the groundbreaking open-source model from Tencent, available now!**  [Explore the original repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld 1.0 Teaser">
</p>

<div align="center">
  <a href="https://3d.hunyuan.tencent.com/sceneTo3D" target="_blank"><img src="https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage" height="22px" alt="Official Site"></a>
  <a href="https://huggingface.co/tencent/HunyuanWorld-1" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg" height="22px" alt="Hugging Face Models"></a>
  <a href="https://3d-models.hunyuan.tencent.com/world/" target="_blank"><img src="https://img.shields.io/badge/Page-bb8a2e.svg?logo=github" height="22px" alt="Model Page"></a>
  <a href="https://arxiv.org/abs/2507.21809" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="22px" alt="Technical Report"></a>
  <a href="https://discord.gg/dNBrdrGGMa" target="_blank"><img src="https://img.shields.io/badge/Discord-white.svg?logo=discord" height="22px" alt="Discord"></a>
  <a href="https://x.com/TencentHunyuan" target="_blank"><img src="https://img.shields.io/badge/Hunyuan-black.svg?logo=x" height="22px" alt="X (Twitter)"></a>
  <a href="#community-resources" target="_blank"><img src="https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore" height="22px" alt="Community"></a>
</div>

## Key Features

*   **Text-to-3D & Image-to-3D Generation:** Transform text descriptions or images into immersive 3D worlds.
*   **360° Panoramic Experience:** Generate immersive experiences through panoramic world proxies.
*   **Mesh Export:** Seamlessly integrate generated worlds into existing computer graphics pipelines.
*   **Interactive & Explorable Worlds:** Disentangled object representations for augmented interactivity.
*   **Quantization Support:** Optimized for consumer-grade GPUs (e.g., 4090) with the HunyuanWorld-1.0-lite version.

## What's New

*   **August 15, 2025:** Released HunyuanWorld-1.0-lite, the quantization version, compatible with consumer-grade GPUs.
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809).
*   **July 26, 2025:** Launched HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model.

> Join the [Discord](https://discord.gg/dNBrdrGGMa) community for discussions and support.

| Community Channels | Links                                                                      |
| :----------------- | :------------------------------------------------------------------------- |
| Wechat             | <img src="assets/qrcode/wechat.png" alt="Wechat QR Code"  height=140>     |
| Xiaohongshu        | <img src="assets/qrcode/xiaohongshu.png" alt="Xiaohongshu QR Code"  height=140> |
| X                  | <img src="assets/qrcode/x.png" alt="X QR Code"  height=140>            |
| Discord            | <img src="assets/qrcode/discord.png" alt="Discord QR Code"  height=140>    |

## ☯️ **HunyuanWorld 1.0: Overview**

### Abstract

HunyuanWorld 1.0 addresses the challenges of generating immersive and playable 3D worlds from text or images.  The model uses a novel framework combining video-based approaches for diversity and 3D-based methods for consistency, to generate interactive, explorable 3D worlds.  Key advantages include 360° immersive experiences via panoramic world proxies, mesh export capabilities for compatibility with existing computer graphics pipelines, and disentangled object representations for augmented interactivity. Extensive experiments demonstrate state-of-the-art performance in creating coherent, explorable, and interactive 3D worlds.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld 1.0 Applications">
</p>

### Architecture

The HunyuanWorld-1.0 generation architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction for high-quality scene-scale 360° 3D world generation from both text and image inputs.

<p align="center">
  <img src="assets/arch.jpg" alt="HunyuanWorld 1.0 Architecture">
</p>

### Performance

HunyuanWorld 1.0 achieves superior results compared to existing methods in text-to-panorama, image-to-panorama, text-to-world, and image-to-world generation.

**Text-to-Panorama Generation:**

| Method           | BRISQUE(↓) | NIQE(↓) | Q-Align(↑) | CLIP-T(↑) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| Diffusion360     | 69.5       | 7.5      | 1.8        | 20.9      |
| MVDiffusion      | 47.9       | 7.1      | 2.4        | 21.5      |
| PanFusion        | 56.6       | 7.6      | 2.2        | 21.0      |
| LayerPano3D      | 49.6       | 6.5      | 3.7        | 21.5      |
| **HunyuanWorld 1.0** | **40.8**    | **5.8**   | **4.4**    | **24.3**   |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(↓) | NIQE(↓) | Q-Align(↑) | CLIP-I(↑) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| Diffusion360     | 71.4       | 7.8      | 1.9        | 73.9      |
| MVDiffusion      | 47.7       | 7.0      | 2.7        | 80.8      |
| **HunyuanWorld 1.0** | **45.2**    | **5.8**   | **4.3**    | **85.1**   |

**Text-to-World Generation:**

| Method           | BRISQUE(↓) | NIQE(↓) | Q-Align(↑) | CLIP-T(↑) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| Director3D       | 49.8       | 7.5      | 3.2        | 23.5      |
| LayerPano3D      | 35.3       | 4.8      | 3.9        | 22.0      |
| **HunyuanWorld 1.0** | **34.6**    | **4.3**   | **4.2**    | **24.0**   |

**Image-to-World Generation:**

| Method           | BRISQUE(↓) | NIQE(↓) | Q-Align(↑) | CLIP-I(↑) |
| ---------------- | ---------- | -------- | ---------- | --------- |
| WonderJourney    | 51.8       | 7.3      | 3.2        | 81.5      |
| DimensionX       | 45.2       | 6.3      | 3.5        | 83.3      |
| **HunyuanWorld 1.0** | **36.2**    | **4.6**   | **3.9**    | **84.5**   |

### Visual Results

<p align="center">
  360 ° immersive and explorable 3D worlds generated by HunyuanWorld 1.0:
</p>

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama Example 1">
</p>

 <p align="left">
  <img src="assets/panorama2.gif" alt="Panorama Example 2">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World Example">
</p>

## 🎁 Models Zoo

The open-source version is based on Flux. Adapt to other image generation models like Hunyuan Image, Kontext, or Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started

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

**Image-to-World Generation:**

```python
# Generate Panorama Image
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create World Scene using Panorama image
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# Result: WORLD SCENE created!
```

**Text-to-World Generation:**

```python
# Generate Panorama Image
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create World Scene using Panorama image
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case7
# Result: WORLD SCENE created!
```

### Quantization & Cache Usage

```python
#Quantization Step 1:
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
#Cache Step 1:
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache

#Quantization Step 2:
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
#Cache Step 2:
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

### Quick Start

```bash
bash scripts/test.sh
```

###  3D World Viewer

Visualize your generated 3D worlds with the included model viewer.

Open `modelviewer.html` in your browser and upload the generated 3D scene files.

<p align="center">
  <img src="assets/quick_look.gif" alt="3D World Viewer Demo">
</p>

## 📑 Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [x] Lite Version
*   [ ] Voyager (RGBD Video Diffusion)

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

## Contact

For any questions, please contact tengfeiwang12@gmail.com.

## Acknowledgements

We thank the contributors to [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their valuable open research.