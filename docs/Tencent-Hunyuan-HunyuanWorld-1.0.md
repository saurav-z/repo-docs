# HunyuanWorld 1.0: Create Interactive 3D Worlds from Text or Images

**Unleash your creativity and transform words or images into explorable, immersive 3D worlds with HunyuanWorld 1.0!**  Dive into a new era of virtual experiences with this cutting-edge, open-source model from Tencent.  Explore the [original repo](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) for the full details.

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://3d.hunyuan.tencent.com/sceneTo3D)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg)](https://huggingface.co/tencent/HunyuanWorld-1)
[![3D Model Page](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github)](https://3d-models.hunyuan.tencent.com/world/)
[![Research Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2507.21809)
[![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord)](https://discord.gg/dNBrdrGGMa)
[![Hunyuan on X](https://img.shields.io/badge/Hunyuan-black.svg?logo=x)](https://x.com/TencentHunyuan)
[![Community Resources](https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore)](#community-resources)

---

## ✨ Key Features

*   **Immersive 360° Worlds:** Experience captivating 3D environments with panoramic world proxies.
*   **Mesh Export Capabilities:** Seamlessly integrate generated worlds with existing 3D graphics pipelines.
*   **Interactive Object Representations:** Enhance interactivity with disentangled object representations.
*   **Text-to-3D & Image-to-3D Generation:** Generate 3D worlds from text prompts or input images.
*   **Open-Source & Accessible:** Leverage open-source code and pre-trained models for easy experimentation.
*   **Optimized for Consumer-Grade GPUs:**  The Lite version enables running on GPUs such as the 4090.
*   **Fast 3D Reconstruction:** Generate 3D content faster with the latest Voyager model.

---

## 📰 News & Updates

*   **September 2, 2025:** Released **HunyuanWorld-Voyager**, an RGB-D Video Diffusion model for 3D-consistent world exploration and fast 3D reconstruction!
*   **August 15, 2025:** Launched **HunyuanWorld-1.0-lite**, the quantized version for improved performance on consumer-grade GPUs.
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809) detailing the innovations behind HunyuanWorld 1.0.
*   **July 26, 2025:**  Introduced the first open-source, simulation-capable, immersive 3D world generation model, **HunyuanWorld-1.0**!

>   Join the **[Discord](https://discord.gg/dNBrdrGGMa)** community to discuss and get support.

---

## ☯️ HunyuanWorld 1.0: Deep Dive

### 💡 Abstract

HunyuanWorld 1.0 addresses the challenge of generating immersive and playable 3D worlds from text or images by combining the strengths of video-based and 3D-based approaches. It leverages 360° panoramic world proxies and a semantically layered 3D mesh representation to create explorable and interactive 3D environments.

<p align="center">
  <img src="assets/teaser.png">
</p>

### 🏗️ Architecture

HunyuanWorld-1.0 integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction for high-quality scene-scale 360° 3D world generation, accepting both text and image inputs.

<p align="center">
  <img src="assets/arch.jpg" alt="HunyuanWorld 1.0 Architecture">
</p>

### 📊 Performance

HunyuanWorld 1.0 achieves state-of-the-art performance in generating coherent, explorable, and interactive 3D worlds.

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| HunyuanWorld 1.0 | **40.8**   | **5.8** | **4.4**    | **24.3**  |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| HunyuanWorld 1.0 | **45.2**   | **5.8** | **4.3**    | **85.1**  |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| HunyuanWorld 1.0 | **34.6**   | **4.3** | **4.2**    | **24.0**  |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| HunyuanWorld 1.0 | **36.2**   | **4.6** | **3.9**    | **84.5**  |

### 🖼️ Visual Results

Experience the results:

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama 1">
</p>

 <p align="left">
  <img src="assets/panorama2.gif" alt="Panorama 2">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World">
</p>

---

## 📦 Models Zoo

Explore the available models.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

---

## 🚀 Get Started

### 🛠️ Environment Setup

```bash
# Instructions from the original readme, updated for better clarity
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml
conda activate HunyuanWorld

# Optional: Real-ESRGAN setup (for upscaling)
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr-fixed
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
cd ..

# Optional: ZIM setup (for scene generation)
git clone https://github.com/naver-ai/ZIM.git
cd ZIM
pip install -e .
cd ..
mkdir zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx
cd ..

# Optional: Draco installation (for mesh compression)
git clone https://github.com/google/draco.git
cd draco
mkdir build
cd build
cmake ..
make
sudo make install
cd ../..

#  Hugging Face login (required to download models)
pip install huggingface_hub
huggingface-cli login
```

### 🎬 Code Usage

**Image to World Generation:**

```python
# 1. Generate a Panorama image from an Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2

# 2. Create a World Scene using the Panorama image.
#    Specify foreground object labels using --labels_fg1 & --labels_fg2.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2

# And then you get your WORLD SCENE!!
```

**Text to World Generation:**

```python
# 1. Generate a Panorama image from a Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7

# 2. Create a World Scene using the Panorama image.
#    Specify foreground object labels using --labels_fg1 & --labels_fg2.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case7

# And then you get your WORLD SCENE!!
```

### 💡 Quantization & Cache Usage

**Image to World Generation (with Optimization):**

```python
# Step 1: Generate Panorama (Quantization and/or Caching)
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention # Quantization
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache # Caching

# Step 2: Generate World Scene (Quantization and/or Caching)
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention # Quantization
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache # Caching
```

**Text to World Generation (with Optimization):**

```python
# Step 1: Generate Panorama (Quantization and/or Caching)
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention  # Quantization
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache # Caching

# Step 2: Generate World Scene (Quantization and/or Caching)
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention # Quantization
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache # Caching
```

### 🚀 Quick Start

```bash
bash scripts/test.sh
```

### 🌐 3D World Viewer

Visualize your generated 3D worlds in your web browser using the provided `modelviewer.html` tool.  Simply upload the generated 3D scene files.

<p align="left">
  <img src="assets/quick_look.gif" alt="Quick Look">
</p>

---

## 🗓️ Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [x] Lite Version
*   [x] Voyager (RGBD Video Diffusion)

---

## 📚 BibTeX

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

---

## 📧 Contact

For any questions, please contact tengfeiwang12@gmail.com.

---

## 🙏 Acknowledgements

We thank the contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their invaluable contributions.