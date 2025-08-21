# HunyuanWorld 1.0: Create Immersive 3D Worlds from Text or Images

**Unleash your creativity and generate explorable, interactive 3D worlds with HunyuanWorld 1.0, the first open-source simulation-capable 3D world generation model.** Explore the official site, Hugging Face models, and more: [HunyuanWorld-1.0 GitHub Repo](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

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

## Key Features

*   **Text-to-3D & Image-to-3D World Generation:**  Transform descriptions or images into interactive 3D environments.
*   **360° Panoramic World Proxies:** Experience immersive worlds with a 360-degree perspective.
*   **Mesh Export Capabilities:** Seamless integration with existing computer graphics pipelines.
*   **Disentangled Object Representations:**  Enables augmented interactivity within the generated worlds.
*   **Quantization & Cache Support**: Optimization for memory and speed, supporting consumer-grade GPUs.

## What's New

*   **August 15, 2025:**  HunyuanWorld-1.0-lite (quantization version) released, now runs on consumer-grade GPUs (e.g., 4090).
*   **July 26, 2025:**  [Technical report](https://arxiv.org/abs/2507.21809) published.
*   **July 26, 2025:**  HunyuanWorld-1.0 released.

>   Join the **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and get help.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> |

## HunyuanWorld 1.0:  Overview

HunyuanWorld 1.0 is a novel framework designed to generate immersive, explorable, and interactive 3D worlds from text and image inputs. It addresses the limitations of existing approaches by combining the strengths of video-based and 3D-based methods.  This is achieved through a semantically layered 3D mesh representation utilizing panoramic images as 360° world proxies for accurate scene decomposition and reconstruction.

<p align="center">
  <img src="assets/application.png">
</p>

## Architecture

HunyuanWorld-1.0 utilizes a sophisticated architecture for high-quality, scene-scale 360° 3D world generation from both text and image inputs. Key components include panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction.

<p align="left">
  <img src="assets/arch.jpg">
</p>

## Performance

HunyuanWorld 1.0 has achieved state-of-the-art results, surpassing other open-source methods in both visual quality and geometric consistency.  The results are broken down by text-to-panorama, image-to-panorama, text-to-world, and image-to-world generation tasks. Here are example results:

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| HunyuanWorld 1.0 | **40.8**              | **5.8**            | **4.4**             | **24.3**           |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| HunyuanWorld 1.0 | **45.2**              | **5.8**            | **4.3**             | **85.1**           |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| HunyuanWorld 1.0 | **34.6**              | **4.3**            | **4.2**             | **24.0**           |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| HunyuanWorld 1.0 | **36.2**              | **4.6**            | **3.9**             | **84.5**           |

## Visual Results

Experience the impressive capabilities of HunyuanWorld 1.0 with these generated 360° immersive 3D worlds:

<p align="left">
  <img src="assets/panorama1.gif">
</p>

 <p align="left">
  <img src="assets/panorama2.gif">
</p>

<p align="left">
  <img src="assets/roaming_world.gif">
</p>

## Model Zoo

Access pre-trained models to get started quickly.  The open-source version is based on Flux, adaptable to image generation models like Hunyuan Image, Kontext, and Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------| 
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## Getting Started

Follow these steps to use HunyuanWorld 1.0:

### Environment Setup

Install dependencies using:

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml
conda activate HunyuanWorld
```

Additional dependencies Real-ESRGAN, ZIM, and Draco are needed as well, see the original README for the commands to install and set them up correctly.

### Code Usage

**Image-to-World Generation:**

```python
# Generate a Panorama image from an Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Create a World Scene with HunyuanWorld 1.0 from the panorama image.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
```

**Text-to-World Generation:**

```python
# Generate a Panorama image from a Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Create a World Scene with HunyuanWorld 1.0 from the panorama image.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
```

### Quantization & Cache Usage
Optimize memory usage and speed up inference with Quantization and Cache. For example:
```python
# Step 1:
# To optimize memory usage and speed up inference, quantization is a practical solution.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2:
# To optimize memory usage and speed up inference, quantization is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

### Quick Start

Run the provided example:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Visualize your generated 3D worlds with the ModelViewer tool.  Open `modelviewer.html` in your browser to upload and explore your 3D scenes.

<p align="left">
  <img src="assets/quick_look.gif">
</p>

## Open-Source Roadmap

*   [x] Inference Code
*   [x] Model Checkpoints
*   [x] Technical Report
*   [x] Lite Version
*   [ ] Voyager (RGBD Video Diffusion)

## Citation

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

For inquiries, contact tengfeiwang12@gmail.com.

## Acknowledgements

The team acknowledges the contributions of  [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen) repositories, for their open research.