# HunyuanWorld 1.0: Generate Interactive 3D Worlds From Text or Images

**Unleash your creativity and explore immersive 3D worlds with HunyuanWorld 1.0, the groundbreaking model that brings your imagination to life.** ([Original Repo](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0))

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld Teaser Image">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px alt="Official Site"></a>
  <a href=https://huggingface.co/tencent/HunyuanWorld-1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px alt="Hugging Face Models"></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px alt="Model Page"></a>
  <a href=https://arxiv.org/abs/2507.21809 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px alt="Research Report"></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px alt="Discord"></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px alt="X"></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px alt="Community"></a>
</div>

## Key Features

*   **Text-to-3D World Generation:** Transform textual descriptions into explorable 3D environments.
*   **Image-to-3D World Generation:** Convert images into immersive 3D scenes.
*   **High-Quality Output:** Generates visually stunning and geometrically consistent 3D worlds.
*   **Panoramic World Proxies:**  Utilizes 360° panoramic images for immersive experiences.
*   **Mesh Export Capabilities:** Compatible with standard computer graphics pipelines.
*   **Interactive Exploration:** Enables interaction with generated objects within the 3D world.
*   **Quantization Support:**  Run on consumer-grade GPUs, such as 4090.
*   **Caching Support:** Speed up inference.

## What's New

*   **August 15, 2025:** Released the quantization version of HunyuanWorld-1.0 (HunyuanWorld-1.0-lite), enabling consumer-grade GPU support.
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809).
*   **July 26, 2025:** Launched HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model.

>  Connect with the community on [Discord](https://discord.gg/dNBrdrGGMa) for discussions and support.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140 alt="Wechat QR Code"> | <img src="assets/qrcode/xiaohongshu.png"  height=140 alt="Xiaohongshu QR Code"> | <img src="assets/qrcode/x.png"  height=140 alt="X QR Code"> | <img src="assets/qrcode/discord.png"  height=140 alt="Discord QR Code"> |

## **HunyuanWorld 1.0: Revolutionizing 3D World Creation**

### Abstract

HunyuanWorld 1.0 tackles the challenge of generating immersive 3D worlds from text or images by combining the strengths of video-based and 3D-based approaches. It overcomes limitations in 3D consistency, rendering efficiency, and data/memory constraints by using a semantically layered 3D mesh representation. Key benefits include 360° immersive experiences, mesh export compatibility, and object-level interactivity. Extensive experiments demonstrate state-of-the-art performance, enabling applications in VR, simulations, game development, and content creation.

<p align="center">
  <img src="assets/application.png" alt="Application Image">
</p>

### Architecture

HunyuanWorld-1.0's architecture seamlessly integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction, enabling scene-scale 360° 3D world generation from both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg" alt="HunyuanWorld Architecture">
</p>

### Performance

HunyuanWorld 1.0 achieves superior performance compared to other open-source panorama and 3D world generation methods.

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 69.5                  | 7.5                | 1.8                 | 20.9               |
| MVDiffusion      | 47.9                  | 7.1                | 2.4                 | 21.5               |
| PanFusion        | 56.6                  | 7.6                | 2.2                 | 21.0               |
| LayerPano3D      | 49.6                  | 6.5                | 3.7                 | 21.5               |
| HunyuanWorld 1.0 | **40.8**              | **5.8**            | **4.4**             | **24.3**           |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 71.4                  | 7.8                | 1.9                 | 73.9               |
| MVDiffusion      | 47.7                  | 7.0                | 2.7                 | 80.8               |
| HunyuanWorld 1.0 | **45.2**              | **5.8**            | **4.3**             | **85.1**           |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Director3D       | 49.8                  | 7.5                | 3.2                 | 23.5               |
| LayerPano3D      | 35.3                  | 4.8                | 3.9                 | 22.0               |
| HunyuanWorld 1.0 | **34.6**              | **4.3**            | **4.2**             | **24.0**           |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| WonderJourney    | 51.8                  | 7.3                | 3.2                 | 81.5               |
| DimensionX       | 45.2                  | 6.3                | 3.5                 | 83.3               |
| HunyuanWorld 1.0 | **36.2**              | **4.6**            | **3.9**             | **84.5**           |

### Visual Results

Experience the immersive power of HunyuanWorld 1.0 through these generated 360° 3D worlds:

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama Example 1">
</p>

<p align="left">
  <img src="assets/panorama2.gif" alt="Panorama Example 2">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World Example">
</p>

## 🎁 Model Zoo

HunyuanWorld 1.0 is based on Flux and can be adapted to other image generation models such as Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started

### Prerequisites
We test our model with Python 3.10 and PyTorch 2.5.0+cu124.

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
# Generate a Panorama image from an image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene with HunyuanWorld 1.0. Indicate foreground object labels
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
```

**Text-to-World Generation:**

```python
# Generate a Panorama image from a prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene with HunyuanWorld 1.0
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case7
```

### Quantization & Cache Usage
```python
#Step 1: Quantization and Cache
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
#Step 2: Generate world scene from panorama with Quantization and Cache
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

```python
#Step 1: Quantization and Cache
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
#Step 2: Generate world scene from panorama with Quantization and Cache
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start

Run the following command for a quick start using the example script:

```bash
bash scripts/test.sh
```

###  3D World Viewer

Use the provided ModelViewer tool to visualize your generated 3D worlds in a web browser.  Open `modelviewer.html` and load your scene files for real-time experiences.

<p align="left">
  <img src="assets/quick_look.gif" alt="Quick Look at the 3D Viewer">
</p>

*Note: Some scenes may fail to load due to hardware limitations.*

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

## Contact

For any questions, please contact tengfeiwang12@gmail.com.

## Acknowledgements

The HunyuanWorld team extends their gratitude to the contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their open research contributions.