# HunyuanWorld 1.0: Create Immersive 3D Worlds from Text and Images

**Unleash your creativity and explore interactive 3D worlds generated from simple text prompts or images with HunyuanWorld 1.0!**  ([Original Repo](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0))

<p align="center">
  <img src="assets/teaser.png">
</p>

## Key Features

*   **Immersive 360° Experiences:** Generate panoramic world proxies for captivating virtual reality and exploration.
*   **Mesh Export Capabilities:** Seamlessly integrate with existing computer graphics pipelines and game development workflows.
*   **Interactive Object Representation:** Experience augmented interactivity within your generated 3D worlds.
*   **Text and Image-to-World Generation:**  Create 3D environments from text prompts or existing images.
*   **High-Quality Results:** Achieves state-of-the-art performance, surpassing baseline methods in visual quality and geometric consistency.
*   **Open-Source and Accessible:** Offering both the model checkpoints, inference code, and a "lite" version.
*   **RGB-D Video Diffusion Model:** HunyuanWorld-Voyager, for 3D-consistent world exploration and fast 3D reconstruction!

## What's New

*   **September 2, 2025:** Released the RGB-D Video Diffusion model [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager/), for 3D-consistent world exploration and fast 3D reconstruction!
*   **August 15, 2025:** Launched a quantization version of HunyuanWorld-1.0 (HunyuanWorld-1.0-lite), optimized to run on consumer-grade GPUs.
*   **July 26, 2025:** Published the [technical report](https://arxiv.org/abs/2507.21809) detailing HunyuanWorld-1.0.
*   **July 26, 2025:** Introduced HunyuanWorld-1.0, an open-source model for creating interactive 3D worlds.

## ☯️ HunyuanWorld 1.0: Dive Deeper

### Abstract

HunyuanWorld 1.0 addresses the challenge of generating immersive and playable 3D worlds from text or images. It combines the strengths of video-based and 3D-based methods, offering rich diversity alongside 3D consistency and efficient rendering. Key advantages include: panoramic proxies for 360° experiences, mesh export for compatibility, and disentangled object representations for interactivity. The core uses a semantically layered 3D mesh representation and panoramic images.  The method enables diverse 3D world generation and versatile applications, including VR, simulation, game development, and content creation.

<p align="center">
  <img src="assets/application.png">
</p>

### Architecture

HunyuanWorld-1.0's architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction. It supports both text and image inputs, ensuring scene-scale 360° 3D world generation.

<p align="left">
  <img src="assets/arch.jpg">
</p>

### Performance

HunyuanWorld 1.0 excels compared to other methods in both text-to- and image-to-panorama and world generation tasks, as demonstrated by benchmark results:

**Text-to-Panorama Generation**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| Diffusion360     | 69.5       | 7.5     | 1.8        | 20.9      |
| MVDiffusion      | 47.9       | 7.1     | 2.4        | 21.5      |
| PanFusion        | 56.6       | 7.6     | 2.2        | 21.0      |
| LayerPano3D      | 49.6       | 6.5     | 3.7        | 21.5      |
| **HunyuanWorld 1.0** | **40.8**   | **5.8** | **4.4**    | **24.3**  |

**Image-to-Panorama Generation**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| Diffusion360     | 71.4       | 7.8     | 1.9        | 73.9      |
| MVDiffusion      | 47.7       | 7.0     | 2.7        | 80.8      |
| **HunyuanWorld 1.0** | **45.2**   | **5.8** | **4.3**    | **85.1**  |

**Text-to-World Generation**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| Director3D       | 49.8       | 7.5     | 3.2        | 23.5      |
| LayerPano3D      | 35.3       | 4.8     | 3.9        | 22.0      |
| **HunyuanWorld 1.0** | **34.6**   | **4.3** | **4.2**    | **24.0**  |

**Image-to-World Generation**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| WonderJourney    | 51.8       | 7.3     | 3.2        | 81.5      |
| DimensionX       | 45.2       | 6.3     | 3.5        | 83.3      |
| **HunyuanWorld 1.0** | **36.2**   | **4.6** | **3.9**    | **84.5**  |

### Visual Results

Experience the results with 360° immersive and explorable 3D worlds:

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

Access the open-source models based on Flux and adaptable to image generation models like Hunyuan Image and Stable Diffusion:

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------|
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started

### Environment Setup

Set up your environment using Python 3.10 and PyTorch 2.5.0+cu124. Then, install the dependencies.

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

**Image-to-World Generation**

```python
# 1. Generate a Panorama from an Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# 2. Create a World Scene using the Panorama.
# Indicate foreground objects labels with --labels_fg1 & --labels_fg2.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 sculptures flowers --labels_fg2 tree mountains --classes outdoor --output_path test_results/case2
# Then you get the WORLD SCENE!
```

**Text-to-World Generation**

```python
# 1. Generate a Panorama from a Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# 2. Create a World Scene using the Panorama.
# Indicate foreground objects labels with --labels_fg1 & --labels_fg2.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# Then you get the WORLD SCENE!
```

### Quantization & Cache Usage

**Quantization & Cache for Image-to-World Generation**

```python
# Step 1: Quantization and/or cache for efficient inference.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2: Use quantization/cache in the scenegen process.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

**Quantization & Cache for Text-to-World Generation**

```python
# Step 1: Quantization and/or cache for efficient inference.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
# Step 2: Use quantization/cache in the scenegen process.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start

Run the example script for a quick demonstration:

```bash
bash scripts/test.sh
```

### 3D World Viewer

Visualize your 3D worlds in your browser using the provided `modelviewer.html` tool.

<p align="left">
  <img src="assets/quick_look.gif">
</p>

*Note: Some scenes may fail to load due to hardware limitations.*

## 📑 Open-Source Plan

*   \[x] Inference Code
*   \[x] Model Checkpoints
*   \[x] Technical Report
*   \[x] Lite Version
*   \[x] Voyager (RGBD Video Diffusion)

## 🔗 BibTeX

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

## Support and Contact

For any questions, please contact tengfeiwang12@gmail.com

## Acknowledgements

The project acknowledges the contributions of the following open-source resources: [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), and [WorldGen](https://github.com/ZiYang-xie/WorldGen).