# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text or Images

**Create breathtaking, explorable 3D worlds from simple text prompts or images with HunyuanWorld 1.0, the cutting-edge AI model from Tencent.**  Explore the [original repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) for more details and to get started.

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld 1.0 Teaser Image">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px alt="Official Site"></a>
  <a href=https://huggingface.co/tencent/HunyuanWorld-1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px alt="Hugging Face Models"></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px alt="Models Page"></a>
  <a href=https://arxiv.org/abs/2507.21809 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px alt="Report"></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px alt="Discord"></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px alt="X/Twitter"></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px alt="Community Resources"></a>
</div>

## Key Features

*   **Text-to-3D and Image-to-3D Generation:** Transform your imagination into interactive 3D environments.
*   **360° Immersive Experiences:** Explore worlds with panoramic views for complete immersion.
*   **Mesh Export:** Seamlessly integrate generated worlds into existing 3D pipelines.
*   **Interactive Elements:** Disentangled object representations allow for dynamic interactions.
*   **High Performance:**  Achieves state-of-the-art results in visual quality and geometric consistency.
*   **Open Source & Accessible:**  Includes model checkpoints, inference code, and a "lite" version for wider compatibility.

## 🔥 What's New

*   **September 2, 2025:** Released [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager/), an RGB-D Video Diffusion model for 3D-consistent world exploration and fast 3D reconstruction.
*   **August 15, 2025:**  Released the quantization version of HunyuanWorld-1.0 (HunyuanWorld-1.0-lite), optimized for consumer-grade GPUs like the 4090.
*   **July 26, 2025:**  Published the [technical report](https://arxiv.org/abs/2507.21809) detailing the architecture and performance of HunyuanWorld-1.0.
*   **July 26, 2025:**  Launched HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model.

> Join the **[Discord](https://discord.gg/dNBrdrGGMa)** community for discussions and support.

| WeChat Group (QR Code)                                     | Xiaohongshu (QR Code)                                           | X                                           | Discord (QR Code)                                           |
|------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------|-------------------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140 alt="Wechat QR Code"> | <img src="assets/qrcode/xiaohongshu.png"  height=140 alt="Xiaohongshu QR Code"> | <img src="assets/qrcode/x.png"  height=140 alt="X/Twitter QR Code"> | <img src="assets/qrcode/discord.png"  height=140 alt="Discord QR Code"> |

## ☯️ **HunyuanWorld 1.0: Technical Overview**

### Abstract

HunyuanWorld 1.0 addresses the challenge of generating immersive and playable 3D worlds from text or images.  It overcomes limitations of existing methods by combining the strengths of video-based and 3D-based approaches. This novel framework generates interactive 3D worlds by utilizing panoramic world proxies, mesh export capabilities, and disentangled object representations.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld 1.0 Application Example">
</p>

### Architecture

HunyuanWorld-1.0's generation architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to achieve high-quality scene-scale 360° 3D world generation, supporting both text and image inputs.

<p align="center">
  <img src="assets/arch.jpg" alt="HunyuanWorld 1.0 Architecture Diagram">
</p>

### Performance

HunyuanWorld 1.0 surpasses existing methods in visual quality and geometric consistency.  The following tables show comparisons with other panorama and 3D world generation methods.  Lower values for metrics like BRISQUE and NIQE are better, while higher values for Q-Align and CLIP are preferred.

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | -------- | ---------- | ---------- |
| Diffusion360     | 69.5       | 7.5      | 1.8        | 20.9       |
| MVDiffusion      | 47.9       | 7.1      | 2.4        | 21.5       |
| PanFusion        | 56.6       | 7.6      | 2.2        | 21.0       |
| LayerPano3D      | 49.6       | 6.5      | 3.7        | 21.5       |
| **HunyuanWorld 1.0** | **40.8**   | **5.8**  | **4.4**    | **24.3**   |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | -------- | ---------- | ---------- |
| Diffusion360     | 71.4       | 7.8      | 1.9        | 73.9       |
| MVDiffusion      | 47.7       | 7.0      | 2.7        | 80.8       |
| **HunyuanWorld 1.0** | **45.2**   | **5.8**  | **4.3**    | **85.1**   |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | -------- | ---------- | ---------- |
| Director3D       | 49.8       | 7.5      | 3.2        | 23.5       |
| LayerPano3D      | 35.3       | 4.8      | 3.9        | 22.0       |
| **HunyuanWorld 1.0** | **34.6**   | **4.3**  | **4.2**    | **24.0**   |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | -------- | ---------- | ---------- |
| WonderJourney    | 51.8       | 7.3      | 3.2        | 81.5       |
| DimensionX       | 45.2       | 6.3      | 3.5        | 83.3       |
| **HunyuanWorld 1.0** | **36.2**   | **4.6**  | **3.9**    | **84.5**   |

### Visual Results

Experience the immersive worlds generated by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama 1 GIF">
</p>

 <p align="left">
  <img src="assets/panorama2.gif" alt="Panorama 2 GIF">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World GIF">
</p>

## 🎁 Models Zoo

HunyuanWorld 1.0 is built on Flux and can be adapted to other image generation models, such as Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Hugging Face                                                                                              |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)          |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)         |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene)     |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)       |

## 🤗 Get Started

Follow these steps to use HunyuanWorld 1.0:

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

**Image-to-World Generation Example:**

```python
# Generate a Panorama image from an image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene using the Panorama image.  Specify foreground object labels.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# View the generated WORLD SCENE!
```

**Text-to-World Generation Example:**

```python
# Generate a Panorama image from a prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene using the Panorama image.  Specify foreground object labels.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# View the generated WORLD SCENE!
```

### Quantization and Cache Usage

Optimize memory and inference speed with quantization and caching:

```python
#  Example with Image-to-World Generation:
#  Step 1: Generate panorama with quantization and/or caching.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2: Generate the scene with quantization and/or caching.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache

#  Example with Text-to-World Generation:
#  Step 1: Generate panorama with quantization and/or caching.
python3 demo_panogen.py --prompt "Your prompt here" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "Your prompt here" --output_path test_results/case7_cache --cache
# Step 2: Generate the scene with quantization and/or caching.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start

Run the example script for a quick start:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Use the provided `modelviewer.html` to visualize your generated 3D worlds in a web browser.  Upload your scene files and explore!

<p align="left">
  <img src="assets/quick_look.gif" alt="3D World Viewer GIF">
</p>

*Note: Scene loading success may vary depending on hardware.*

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

## Contact

For questions, please contact tengfeiwang12@gmail.com.

## Acknowledgements

The developers of HunyuanWorld 1.0 would like to thank the open source community and contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their open research.