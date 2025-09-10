# HunyuanWorld 1.0: Create Immersive 3D Worlds From Text or Images

HunyuanWorld 1.0 allows you to generate interactive and explorable 3D worlds from text descriptions or images.  Explore the cutting edge of 3D world generation with this innovative model!  [Visit the original repository for more details.](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://3d.hunyuan.tencent.com/sceneTo3D)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg)](https://huggingface.co/tencent/HunyuanWorld-1)
[![3D Models Page](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github)](https://3d-models.hunyuan.tencent.com/world/)
[![Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2507.21809)
[![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord)](https://discord.gg/dNBrdrGGMa)
[![Hunyuan on X](https://img.shields.io/badge/Hunyuan-black.svg?logo=x)](https://x.com/TencentHunyuan)
[![Community Resources](https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore)](#community-resources)

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld 1.0 Teaser Image">
</p>

> "To see a World in a Grain of Sand, and a Heaven in a Wild Flower"

## 🔥 **Key Updates & News**

*   **September 2, 2025:** Released [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager/), an RGB-D Video Diffusion model for 3D-consistent world exploration and fast 3D reconstruction.
*   **August 15, 2025:** Launched HunyuanWorld-1.0-lite, a quantized version for use on consumer-grade GPUs.
*   **July 26, 2025:**  Published the [technical report](https://arxiv.org/abs/2507.21809) detailing HunyuanWorld-1.0.
*   **July 26, 2025:**  Initial release of HunyuanWorld-1.0, a simulation-capable, immersive 3D world generation model.

Join our [Discord](https://discord.gg/dNBrdrGGMa) for discussions and support.

<table align="center">
  <tr>
    <td><img src="assets/qrcode/wechat.png"  height=140></td>
    <td><img src="assets/qrcode/xiaohongshu.png"  height=140></td>
    <td><img src="assets/qrcode/x.png"  height=140></td>
    <td><img src="assets/qrcode/discord.png"  height=140></td>
  </tr>
</table>

## 🔑 **Key Features of HunyuanWorld 1.0**

*   **360° Immersive Experiences:** Generate panoramic world proxies for a fully immersive view.
*   **Mesh Export Capabilities:**  Seamlessly integrate with existing computer graphics pipelines.
*   **Disentangled Object Representations:** Enables augmented interactivity within the generated worlds.

## ☯️ **HunyuanWorld 1.0: Overview**

HunyuanWorld 1.0 tackles the challenge of generating playable 3D worlds from text or images. The framework combines the strengths of video-based and 3D-based approaches, providing both rich diversity and geometric consistency.

### **Abstract**

HunyuanWorld 1.0 offers a novel approach to 3D world generation from text and image inputs.  It overcomes limitations of existing methods by:

*   Creating 360° immersive experiences
*   Enabling mesh export
*   Providing disentangled object representations for enhanced interactivity.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld 1.0 Application Examples">
</p>

### **Architecture**

The core of HunyuanWorld-1.0 utilizes a semantically layered 3D mesh representation leveraging panoramic images as 360° world proxies for effective scene decomposition and reconstruction. This allows for the generation of diverse 3D worlds from text and image inputs.

<p align="center">
  <img src="assets/arch.jpg" alt="HunyuanWorld 1.0 Architecture Diagram">
</p>

### **Performance**

HunyuanWorld 1.0 surpasses other open-source panorama and 3D world generation methods in visual quality and geometric consistency.

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

### **Visual Results**

Experience the immersive 3D worlds generated by HunyuanWorld 1.0:

<p align="center">
  <img src="assets/panorama1.gif" alt="Panorama Example 1">
</p>

<p align="center">
  <img src="assets/panorama2.gif" alt="Panorama Example 2">
</p>

<p align="center">
  <img src="assets/roaming_world.gif" alt="Roaming World Example">
</p>

## 📦 **Model Zoo**

HunyuanWorld 1.0 is based on Flux and can be easily adapted to other image generation models.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🚀 **Getting Started with HunyuanWorld 1.0**

Follow the steps below to use HunyuanWorld 1.0:

### **Environment Setup**

The model was tested with Python 3.10 and PyTorch 2.5.0+cu124.

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

### **Code Usage**

**Image-to-World Generation:**

```python
# Generate a Panorama image from an Image
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene using the Panorama image.
# Specify foreground object labels using --labels_fg1 & --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# View your WORLD SCENE!
```

**Text-to-World Generation:**

```python
# Generate a Panorama image from A Prompt
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene using the Panorama image.
# Specify foreground object labels using --labels_fg1 & --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# View your WORLD SCENE!
```

### **Quantization & Cache Usage**

**Image-to-World Generation (Quantization/Cache):**

```python
# Step 1:  Quantization and Cache for Optimization.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2:  Quantization and Cache for Optimization.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

**Text-to-World Generation (Quantization/Cache):**

```python
# Step 1:  Quantization and Cache for Optimization.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
# Step 2:  Quantization and Cache for Optimization.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### **Quick Start**

Explore more examples in the ```examples``` directory by running:

```bash
bash scripts/test.sh
```

### **3D World Viewer**

Use the provided ModelViewer tool to visualize your 3D worlds in a web browser. Open ```modelviewer.html```, upload the generated 3D scene files, and explore!

<p align="center">
  <img src="assets/quick_look.gif" alt="3D World Viewer Example">
</p>

*Note:  Scene loading success depends on hardware capabilities.*

## 📑 **Open-Source Roadmap**

*   \[x] Inference Code
*   \[x] Model Checkpoints
*   \[x] Technical Report
*   \[x] Lite Version
*   \[x] Voyager (RGBD Video Diffusion)

## 🔗 **BibTeX**

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

## 💬 **Contact**

For any questions, please contact tengfeiwang12@gmail.com.

## 🙏 **Acknowledgements**

We thank the contributors to the following open-source projects: [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen).