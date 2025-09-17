# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text or Images

HunyuanWorld 1.0 is a groundbreaking framework for creating immersive, explorable, and interactive 3D worlds from text and images, offering unparalleled realism and user experience.  Explore the [original repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) for more details.

<p align="center">
  <img src="assets/teaser.png">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HunyuanWorld-1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://arxiv.org/abs/2507.21809 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

## Key Features

*   **360° Immersive Experiences:** Generate panoramic world proxies for a fully immersive experience.
*   **Mesh Export:** Compatible with existing computer graphics pipelines through mesh export capabilities.
*   **Disentangled Object Representations:** Enables augmented interactivity through disentangled object representations.
*   **Text-to-World & Image-to-World:** Generate scenes from both textual descriptions and images.
*   **High Visual Quality and Geometric Consistency:** State-of-the-art performance demonstrated in generating coherent, explorable, and interactive 3D worlds.
*   **Fast 3D Reconstruction:** Supports 3D-consistency world exploration and fast 3D reconstruction.
*   **Quantization Support:** Provides support for running on consumer-grade GPUs.

## What's New

*   **September 2, 2025:** Released [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager/), an RGB-D Video Diffusion model supporting 3D-consistency world exploration and fast 3D reconstruction.
*   **August 15, 2025:** Released a quantization version of HunyuanWorld-1.0 (HunyuanWorld-1.0-lite), supporting consumer-grade GPUs.
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809).
*   **July 26, 2025:** Launched HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model.

## Architecture

HunyuanWorld-1.0 integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to generate high-quality, scene-scale 360° 3D worlds from text or image inputs.

<p align="left">
  <img src="assets/arch.jpg">
</p>

## Performance

HunyuanWorld 1.0 outperforms other methods in visual quality and geometric consistency, as demonstrated by the following results:

### Text-to-Panorama Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 69.5                  | 7.5                | 1.8                 | 20.9               |
| MVDiffusion      | 47.9                  | 7.1                | 2.4                 | 21.5               |
| PanFusion        | 56.6                  | 7.6                | 2.2                 | 21.0               |
| LayerPano3D      | 49.6                  | 6.5                | 3.7                 | 21.5               |
| HunyuanWorld 1.0 | **40.8**              | **5.8**            | **4.4**             | **24.3**           |

### Image-to-Panorama Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 71.4                  | 7.8                | 1.9                 | 73.9               |
| MVDiffusion      | 47.7                  | 7.0                | 2.7                 | 80.8               |
| HunyuanWorld 1.0 | **45.2**              | **5.8**            | **4.3**             | **85.1**           |

### Text-to-World Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Director3D       | 49.8                  | 7.5                | 3.2                 | 23.5               |
| LayerPano3D      | 35.3                  | 4.8                | 3.9                 | 22.0               |
| HunyuanWorld 1.0 | **34.6**              | **4.3**            | **4.2**             | **24.0**           |

### Image-to-World Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| WonderJourney    | 51.8                  | 7.3                | 3.2                 | 81.5               |
| DimensionX       | 45.2                  | 6.3                | 3.5                 | 83.3               |
| HunyuanWorld 1.0 | **36.2**              | **4.6**            | **3.9**             | **84.5**           |

## Visual Results

See the impressive 360° immersive and explorable 3D worlds generated by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif">
</p>

 <p align="left">
  <img src="assets/panorama2.gif">
</p> 

<p align="left">
  <img src="assets/roaming_world.gif">
</p>

## Models Zoo

The open-source version of HY World 1.0 is based on Flux and can be adapted to other image generation models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------| 
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## Getting Started

Follow the steps below to get started with HunyuanWorld 1.0:

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
# Generate a Panorama image from an Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene from the Panorama image.  Specify foreground object labels using labels_fg1 & labels_fg2.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# The generated WORLD SCENE is ready!
```

*   **Text-to-World Generation:**

```python
# Generate a Panorama image from a Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene from the Panorama image, specifying foreground object labels if needed.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# The generated WORLD SCENE is ready!
```

### Quantization and Cache Usage

*   **Image-to-World Generation with Quantization/Cache:**

```python
# Step 1:  Quantization
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2: Quantization
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

*   **Text-to-World Generation with Quantization/Cache:**

```python
# Step 1: Quantization
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
# Step 2: Quantization
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start

Run the provided example:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Visualize your generated 3D worlds using the ModelViewer tool. Open `modelviewer.html` in your browser and load your scene files.

<p align="left">
  <img src="assets/quick_look.gif">
</p>

*Note: Some scenes might fail to load due to hardware limitations.*

## Open-Source Plan

*   \[x] Inference Code
*   \[x] Model Checkpoints
*   \[x] Technical Report
*   \[x] Lite Version
*   \[x] Voyager (RGBD Video Diffusion)

## 📑 BibTeX

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

We thank the contributors to [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen) repositories for their open research.