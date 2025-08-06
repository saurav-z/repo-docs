# HunyuanWorld-1.0: Generate Immersive 3D Worlds from Text or Images

HunyuanWorld-1.0 allows you to generate interactive and explorable 3D worlds from text descriptions or images, offering a new dimension in virtual content creation.  Explore the [original repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) for more details.

## Key Features:

*   **Text-to-3D and Image-to-3D Generation:** Create 3D worlds from textual descriptions or by using an image as input.
*   **Immersive 360° Experiences:**  Experience generated worlds with full 360-degree panoramic views, creating an immersive virtual environment.
*   **Mesh Export Capabilities:**  Export generated worlds as meshes, allowing seamless integration with existing computer graphics pipelines for further editing and customization.
*   **Interactive Object Representations:**  Objects within the generated worlds are represented in a disentangled manner, enabling augmented interactivity with the virtual environment.
*   **State-of-the-Art Performance:**  Achieves superior visual quality and geometric consistency compared to other methods.

## Quick Links

*   **Official Site:** [![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://3d.hunyuan.tencent.com/sceneTo3D)
*   **Hugging Face Models:** [![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg)](https://huggingface.co/tencent/HunyuanWorld-1)
*   **3D Models:** [![3D Models](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github)](https://3d-models.hunyuan.tencent.com/world/)
*   **Technical Report:** [![Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2507.21809)
*   **Discord:** [![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord)](https://discord.gg/dNBrdrGGMa)
*   **X (Twitter):** [![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x)](https://x.com/TencentHunyuan)
*   **Community Resources:** [![Community](https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore)](#community-resources)

## News

*   **July 26, 2025:** Released the technical report, details available [here](https://arxiv.org/abs/2507.21809).
*   **July 26, 2025:** Launched HunyuanWorld-1.0, the first open-source simulation-capable 3D world generation model!

Join our [Discord](https://discord.gg/dNBrdrGGMa) to discuss and get help.

## ☯️ **HunyuanWorld 1.0 Overview**

HunyuanWorld 1.0 leverages a novel framework to generate immersive, explorable, and interactive 3D worlds from text and image inputs. The core of the framework lies in its semantically layered 3D mesh representation.  This representation uses panoramic images as 360° world proxies for semantic-aware world decomposition and reconstruction.

### Architecture

<p align="left">
  <img src="assets/arch.jpg">
</p>

### Performance

HunyuanWorld 1.0 surpasses existing methods in generating high-quality 3D worlds, as evidenced by the benchmark results.

Text-to-panorama generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 69.5                  | 7.5                | 1.8                 | 20.9               |
| MVDiffusion      | 47.9                  | 7.1                | 2.4                 | 21.5               |
| PanFusion        | 56.6                  | 7.6                | 2.2                 | 21.0               |
| LayerPano3D      | 49.6                  | 6.5                | 3.7                 | 21.5               |
| HunyuanWorld 1.0 | **40.8**              | **5.8**            | **4.4**             | **24.3**           |

Image-to-panorama generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 71.4                  | 7.8                | 1.9                 | 73.9               |
| MVDiffusion      | 47.7                  | 7.0                | 2.7                 | 80.8               |
| HunyuanWorld 1.0 | **45.2**              | **5.8**            | **4.3**             | **85.1**           |

Text-to-world generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Director3D       | 49.8                  | 7.5                | 3.2                 | 23.5               |
| LayerPano3D      | 35.3                  | 4.8                | 3.9                 | 22.0               |
| HunyuanWorld 1.0 | **34.6**              | **4.3**            | **4.2**             | **24.0**           |

Image-to-world generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| WonderJourney    | 51.8                  | 7.3                | 3.2                 | 81.5               |
| DimensionX       | 45.2                  | 6.3                | 3.5                 | 83.3               |
| HunyuanWorld 1.0 | **36.2**              | **4.6**            | **3.9**             | **84.5**           |

### Visual Results

Experience some of the 3D worlds generated by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif">
</p>

 <p align="left">
  <img src="assets/panorama2.gif">
</p>

<p align="left">
  <img src="assets/roaming_world.gif">
</p>

## 🎁 Models Zoo

The open-source version is based on Flux, and the method can be easily adapted to other image generation models such as Hunyuan Image, Kontext, Stable Diffusion.

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

**Image to World Generation:**

```python
# Generate Panorama
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Generate World Scene
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2
```

**Text to World Generation:**

```python
# Generate Panorama
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Generate World Scene
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
```

### Quick Start

Run the example script for a quick start:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Use the provided `modelviewer.html` file to view your generated 3D worlds in a web browser.  Simply open the HTML file and load your 3D scene files.

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

For any questions, please contact tengfeiwang12@gmail.com.

## Acknowledgements

We thank the contributors to the following projects: [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen).