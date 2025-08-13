<!-- Improved & SEO-Optimized README for HunyuanWorld-1.0 -->

# HunyuanWorld-1.0: Create Immersive 3D Worlds from Text and Images

**Unleash your imagination and bring your ideas to life with HunyuanWorld-1.0, a groundbreaking model for generating interactive and explorable 3D worlds.**  Explore the original repository at: [https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)

<p align="center">
  <img src="assets/teaser.png" alt="HunyuanWorld-1.0 Teaser Image">
</p>

<div align="center">
  <a href="https://3d.hunyuan.tencent.com/sceneTo3D" target="_blank"><img src="https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage" height="22px" alt="Official Site"></a>
  <a href="https://huggingface.co/tencent/HunyuanWorld-1" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg" height="22px" alt="Hugging Face Models"></a>
  <a href="https://3d-models.hunyuan.tencent.com/world/" target="_blank"><img src="https://img.shields.io/badge/Page-bb8a2e.svg?logo=github" height="22px" alt="Model Page"></a>
  <a href="https://arxiv.org/abs/2507.21809" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="22px" alt="Research Report"></a>
  <a href="https://discord.gg/dNBrdrGGMa" target="_blank"><img src="https://img.shields.io/badge/Discord-white.svg?logo=discord" height="22px" alt="Discord"></a>
  <a href="https://x.com/TencentHunyuan" target="_blank"><img src="https://img.shields.io/badge/Hunyuan-black.svg?logo=x" height="22px" alt="X (Twitter)"></a>
  <a href="#community-resources" target="_blank"><img src="https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore" height="22px" alt="Community Resources"></a>
</div>

<br>

## Key Features of HunyuanWorld-1.0

*   **Text-to-3D & Image-to-3D Generation:**  Effortlessly transform text descriptions or images into immersive 3D worlds.
*   **360° Panoramic Experience:** Generate fully immersive 360-degree environments.
*   **Mesh Export:**  Export generated scenes in mesh format for use in existing computer graphics pipelines.
*   **Interactive and Explorable Worlds:**  Create dynamic 3D environments that users can explore and interact with.
*   **State-of-the-Art Performance:**  Achieves superior visual quality and geometric consistency compared to other methods.

## What's New

*   **July 26, 2025:** [Technical Report](https://arxiv.org/abs/2507.21809) released, detailing the inner workings of HunyuanWorld-1.0.
*   **July 26, 2025:**  The first open-source model for simulation-capable, immersive 3D world generation, HunyuanWorld-1.0, is now available.

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** community to discuss and receive support.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> |

## ☯️ HunyuanWorld 1.0: The Future of 3D World Creation

### Abstract

HunyuanWorld-1.0 tackles the challenge of creating immersive, interactive 3D worlds from text or images.  It overcomes limitations of existing methods by combining the benefits of video-based and 3D-based approaches. Key innovations include:

*   **Panoramic World Proxies:**  360° immersive experiences using panoramic representations.
*   **Semantic-Aware 3D Mesh Representation:**  Enables world decomposition and reconstruction.
*   **Disentangled Object Representations:**  Enhances interactivity within the generated worlds.

This framework allows for versatile applications in virtual reality, game development, and content creation.

<p align="center">
  <img src="assets/application.png" alt="HunyuanWorld-1.0 Application">
</p>

### Architecture

The HunyuanWorld-1.0 architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction.

<p align="left">
  <img src="assets/arch.jpg" alt="HunyuanWorld-1.0 Architecture">
</p>

### Performance

HunyuanWorld-1.0 demonstrates superior performance compared to other open-source methods in both text-to-panorama/world and image-to-panorama/world generation, as shown by the following metrics:

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | :--------: | :------: | :--------: | :--------: |
| Diffusion360     |    69.5    |   7.5    |    1.8     |    20.9    |
| MVDiffusion      |    47.9    |   7.1    |    2.4     |    21.5    |
| PanFusion        |    56.6    |   7.6    |    2.2     |    21.0    |
| LayerPano3D      |    49.6    |   6.5    |    3.7     |    21.5    |
| **HunyuanWorld 1.0** |   **40.8**   |   **5.8**   |   **4.4**    |   **24.3**   |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | :--------: | :------: | :--------: | :--------: |
| Diffusion360     |    71.4    |   7.8    |    1.9     |    73.9    |
| MVDiffusion      |    47.7    |   7.0    |    2.7     |    80.8    |
| **HunyuanWorld 1.0** |   **45.2**   |   **5.8**   |   **4.3**    |   **85.1**   |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | :--------: | :------: | :--------: | :--------: |
| Director3D       |    49.8    |   7.5    |    3.2     |    23.5    |
| LayerPano3D      |    35.3    |   4.8    |    3.9     |    22.0    |
| **HunyuanWorld 1.0** |   **34.6**   |   **4.3**   |   **4.2**    |   **24.0**   |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | :--------: | :------: | :--------: | :--------: |
| WonderJourney    |    51.8    |   7.3    |    3.2     |    81.5    |
| DimensionX       |    45.2    |   6.3    |    3.5     |    83.3    |
| **HunyuanWorld 1.0** |   **36.2**   |   **4.6**   |   **3.9**    |   **84.5**   |

### Visual Results

Experience the immersive 3D worlds generated by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif" alt="Panorama 1 GIF">
</p>

<p align="left">
  <img src="assets/panorama2.gif" alt="Panorama 2 GIF">
</p>

<p align="left">
  <img src="assets/roaming_world.gif" alt="Roaming World GIF">
</p>

## 🎁 Model Zoo

HunyuanWorld-1.0 is based on Flux and can be adapted to other image generation models.

| Model                          | Description                 | Date       | Size  | Hugging Face                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started with HunyuanWorld 1.0

Follow these steps to start using HunyuanWorld 1.0:

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
# Generate a Panorama image from an image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene using the Panorama image.
# Indicate foreground objects using --labels_fg1 and --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# View your WORLD SCENE!
```

**Text to World Generation:**

```python
# Generate a Panorama image from a prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene using the Panorama image.
# Indicate foreground objects using --labels_fg1 and --labels_fg2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# View your WORLD SCENE!
```

### Quick Start

Run the quick start example:
```bash
bash scripts/test.sh
```

### 3D World Viewer

Use the ModelViewer tool to visualize your 3D worlds in a web browser:

1.  Open `modelviewer.html` in your browser.
2.  Upload the generated 3D scene files.
3.  Enjoy the real-time interactive experience.

<p align="left">
  <img src="assets/quick_look.gif" alt="Quick Look at 3D World Viewer">
</p>

*Note: Some scenes may not load due to hardware limitations.*

## 📑 Open-Source Plan

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [ ] TensorRT Version (Planned)
*   [ ] RGBD Video Diffusion (Planned)

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

For any questions, please email tengfeiwang12@gmail.com.

## Acknowledgements

Special thanks to the contributors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen) for their open-source contributions.