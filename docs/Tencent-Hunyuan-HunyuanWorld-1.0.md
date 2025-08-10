<!-- Improved README.md -->
# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text and Images

**Unleash your imagination and bring your visions to life by creating explorable, interactive 3D worlds with [HunyuanWorld 1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0), a groundbreaking model from Tencent.**

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld Teaser">
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
<br>

> *"To see a World in a Grain of Sand, and a Heaven in a Wild Flower"* - William Blake, echoed in HunyuanWorld's ability to generate detailed 3D environments.

<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/blob/747b3e41-df9c-4cd2-b1d1-c0dce63f63ef" alt="HunyuanWorld Example">
</p>

## Key Features

*   **Immersive 360° Experiences:** Generate panoramic world proxies for a truly immersive experience.
*   **Mesh Export Capabilities:** Seamless integration with existing computer graphics pipelines.
*   **Interactive 3D Worlds:** Disentangled object representations enable augmented interactivity.
*   **Text and Image Input:** Create 3D worlds from text prompts or existing images.
*   **State-of-the-Art Performance:** Achieve superior visual quality and geometric consistency compared to other methods.

## What's New

*   **July 26, 2025:** HunyuanWorld-1.0's [technical report](https://arxiv.org/abs/2507.21809) released, detailing the innovative framework.
*   **July 26, 2025:** Public release of HunyuanWorld-1.0, an open-source model for 3D world generation.

## ☯️ **HunyuanWorld 1.0 Overview**

HunyuanWorld 1.0 is a novel framework designed to overcome the limitations of existing 3D world generation methods. It combines the strengths of both video-based and 3D-based approaches, resulting in the creation of immersive, explorable, and interactive 3D worlds from text and image inputs. This is achieved through a semantically layered 3D mesh representation that utilizes panoramic images as 360° world proxies.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld Applications">
</p>

### Architecture

The generation architecture of Tencent HunyuanWorld-1.0 integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to produce high-quality, scene-scale 360° 3D worlds from text and image inputs.

<p align="center">
  <img src="assets/arch.jpg" alt="HunyuanWorld Architecture">
</p>

### Performance

HunyuanWorld 1.0 demonstrates improved performance in visual quality and geometric consistency compared to existing methods, as evidenced by the following benchmarks:

#### Text-to-Panorama Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | :--------: | :------: | :--------: | :-------: |
| Diffusion360     |    69.5    |   7.5    |    1.8     |   20.9    |
| MVDiffusion      |    47.9    |   7.1    |    2.4     |   21.5    |
| PanFusion        |    56.6    |   7.6    |    2.2     |   21.0    |
| LayerPano3D      |    49.6    |   6.5    |    3.7     |   21.5    |
| HunyuanWorld 1.0 |   **40.8**   |  **5.8**   |   **4.4**    |  **24.3**   |

#### Image-to-Panorama Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | :--------: | :------: | :--------: | :-------: |
| Diffusion360     |    71.4    |   7.8    |    1.9     |   73.9    |
| MVDiffusion      |    47.7    |   7.0    |    2.7     |   80.8    |
| HunyuanWorld 1.0 |   **45.2**   |  **5.8**   |   **4.3**    |  **85.1**   |

#### Text-to-World Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | :--------: | :------: | :--------: | :-------: |
| Director3D       |    49.8    |   7.5    |    3.2     |   23.5    |
| LayerPano3D      |    35.3    |   4.8    |    3.9     |   22.0    |
| HunyuanWorld 1.0 |   **34.6**   |  **4.3**   |   **4.2**    |  **24.0**   |

#### Image-to-World Generation

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | :--------: | :------: | :--------: | :-------: |
| WonderJourney    |    51.8    |   7.3    |    3.2     |   81.5    |
| DimensionX       |    45.2    |   6.3    |    3.5     |   83.3    |
| HunyuanWorld 1.0 |   **36.2**   |  **4.6**   |   **3.9**    |  **84.5**   |

### Visual Results

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

## 🎁 Models Zoo

The open-source version of HunyuanWorld 1.0 is based on Flux and can be easily adapted to other image generation models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Hugging Face Link                                                                                     |
|--------------------------------|-----------------------------|------------|-------|-----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)       |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)      |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started

Follow the steps below to begin using HunyuanWorld 1.0:

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

*   **Image to World Generation:**

    ```python
    # Generate a Panorama image.
    python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
    # Create a World Scene from the Panorama image.  Specify foreground object labels (labels_fg1, labels_fg2).
    CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
    ```

*   **Text to World Generation:**

    ```python
    # Generate a Panorama image.
    python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
    # Create a World Scene from the Panorama image. Specify foreground object labels (labels_fg1, labels_fg2).
    CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
    ```

### Quick Start

Run the examples:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Visualize your generated 3D worlds in your web browser using the provided ModelViewer tool.
Open `modelviewer.html` and upload the generated 3D scene files.

<p align="center">
  <img src="assets/quick_look.gif" alt="Quick Look">
</p>

*Note: Scene loading may be limited by hardware.*

## 📑 Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [ ] TensorRT Version (Coming Soon)
*   [ ] RGBD Video Diffusion (Future Development)

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

## Community and Support

*   Join the community on [Discord](https://discord.gg/dNBrdrGGMa).
*   Stay updated on [X (Twitter)](https://x.com/TencentHunyuan).

<p align="center">
    <img src="assets/qrcode/wechat.png" alt="Wechat QR Code"  height=140>
    <img src="assets/qrcode/xiaohongshu.png" alt="Xiaohongshu QR Code"  height=140>
    <img src="assets/qrcode/x.png" alt="X QR Code"  height=140>
    <img src="assets/qrcode/discord.png" alt="Discord QR Code"  height=140>
</p>

## Contact

For any questions or inquiries, please reach out via email: [tengfeiwang12@gmail.com](mailto:tengfeiwang12@gmail.com)

## Acknowledgements

This project leverages the advancements of several open-source repositories, including [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen). We extend our gratitude to all contributors for their valuable research contributions.