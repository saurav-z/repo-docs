# HunyuanWorld 1.0: Create Immersive 3D Worlds from Text and Images

**Explore the future of 3D world generation with HunyuanWorld 1.0, a cutting-edge model from Tencent, capable of transforming text and images into explorable and interactive 3D environments.**  [View the original repository on GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://3d.hunyuan.tencent.com/sceneTo3D)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg)](https://huggingface.co/tencent/HunyuanWorld-1)
[![Page](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github)](https://3d-models.hunyuan.tencent.com/world/)
[![Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2507.21809)
[![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord)](https://discord.gg/dNBrdrGGMa)
[![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x)](https://x.com/TencentHunyuan)
[![Community](https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore)](#community-resources)

<br>

## Key Features of HunyuanWorld 1.0:

*   **Text-to-3D and Image-to-3D Generation:** Transform text descriptions or images into detailed 3D worlds.
*   **Immersive 360° Experiences:** Enjoy fully panoramic and explorable environments.
*   **Mesh Export Compatibility:** Seamless integration with existing computer graphics pipelines.
*   **Interactive Object Representation:**  Disentangled object representations for enhanced user interaction.
*   **State-of-the-Art Performance:** Achieves superior visual quality and geometric consistency compared to other methods.
*   **Versatile Applications:** Ideal for virtual reality, game development, and interactive content creation.

## 🔥 What's New

*   **July 26, 2025:** The [technical report](https://arxiv.org/abs/2507.21809) for HunyuanWorld-1.0 is released.
*   **July 26, 2025:**  HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model, is available!

Join the community:  **[Discord](https://discord.gg/dNBrdrGGMa)**

## ☯️ **HunyuanWorld 1.0: Overview**

HunyuanWorld 1.0 sets a new standard in generating immersive and interactive 3D worlds. It tackles the challenges of creating high-quality 3D environments from text and images by combining the strengths of video-based and 3D-based approaches. It utilizes a novel framework featuring a semantically layered 3D mesh representation that leverages panoramic images as 360° world proxies for semantic-aware world decomposition and reconstruction, enabling the generation of diverse 3D worlds.

<p align="center">
  <img src="assets/application.png">
</p>

### Architecture

HunyuanWorld-1.0 integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction for creating high-quality, scene-scale 360° 3D worlds from both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg">
</p>

### Performance

HunyuanWorld 1.0 significantly outperforms existing methods in visual quality and geometric consistency.  Here's a comparison of performance metrics:

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| Diffusion360     | 69.5       | 7.5     | 1.8        | 20.9      |
| MVDiffusion      | 47.9       | 7.1     | 2.4        | 21.5      |
| PanFusion        | 56.6       | 7.6     | 2.2        | 21.0      |
| LayerPano3D      | 49.6       | 6.5     | 3.7        | 21.5      |
| **HunyuanWorld 1.0** | **40.8**   | **5.8** | **4.4**    | **24.3**  |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| Diffusion360     | 71.4       | 7.8     | 1.9        | 73.9      |
| MVDiffusion      | 47.7       | 7.0     | 2.7        | 80.8      |
| **HunyuanWorld 1.0** | **45.2**   | **5.8** | **4.3**    | **85.1**  |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| Director3D       | 49.8       | 7.5     | 3.2        | 23.5      |
| LayerPano3D      | 35.3       | 4.8     | 3.9        | 22.0      |
| **HunyuanWorld 1.0** | **34.6**   | **4.3** | **4.2**    | **24.0**  |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| WonderJourney    | 51.8       | 7.3     | 3.2        | 81.5      |
| DimensionX       | 45.2       | 6.3     | 3.5        | 83.3      |
| **HunyuanWorld 1.0** | **36.2**   | **4.6** | **3.9**    | **84.5**  |

### Visual Results

Experience the impressive 360° immersive 3D worlds generated by HunyuanWorld 1.0:

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

The open-source version of HY World 1.0 is built on Flux and can be easily adapted to other image generation models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started

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
# Generate a Panorama image from an Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene from the Panorama image using HunyuanWorld 1.0.
# Indicate the foreground object labels with --labels_fg1 & --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# View your generated WORLD SCENE!
```

**Text to World Generation:**

```python
# Generate a Panorama image from a Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene from the Panorama image using HunyuanWorld 1.0.
# Indicate the foreground object labels with --labels_fg1 & --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# View your generated WORLD SCENE!
```

### Quick Start

Run the example:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Use the ModelViewer tool to visualize your 3D worlds in a web browser. Open `modelviewer.html`, upload the generated 3D scene files, and experience real-time play.

<p align="left">
  <img src="assets/quick_look.gif">
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

## 📞 Contact

For any questions, please contact: tengfeiwang12@gmail.com.

## Acknowledgements

This project acknowledges the contributions of the [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen) repositories.