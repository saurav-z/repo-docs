# HunyuanWorld 1.0: Generate Immersive 3D Worlds from Text or Images

**Create stunning, explorable 3D worlds with ease using HunyuanWorld 1.0, the cutting-edge model from Tencent, leveraging the power of AI to transform your creative visions into reality.**  [Explore the original repository on GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

[<img src="assets/teaser.png" alt="HunyuanWorld 1.0 Teaser" width="500">](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)

**Key Features:**

*   ✨ **Generate 360° Immersive Experiences:** Explore detailed 3D worlds with panoramic world proxies.
*   ⚙️ **Mesh Export Capabilities:** Seamlessly integrate generated worlds into existing graphics pipelines.
*   🧩 **Disentangled Object Representations:** Enjoy augmented interactivity with easily manipulable objects.
*   ⚡ **Quantization Support:** Run the model on consumer-grade GPUs with the "lite" version.
*   🚀 **Text and Image Input:** Create worlds from text prompts or existing images.

**Key Highlights:**

*   **State-of-the-Art Performance:** Achieve superior visual quality and geometric consistency compared to other methods.
*   **Versatile Applications:** Perfect for virtual reality, physical simulation, game development, and interactive content creation.
*   **Open Source:**  All inference code, model checkpoints, and the technical report are available to the public.

---

## 🚀 **What's New?**

*   **August 15, 2025:**  HunyuanWorld-1.0-lite (quantization version) released, supports consumer-grade GPUs like the 4090!
*   **July 26, 2025:**  [Technical Report](https://arxiv.org/abs/2507.21809) of HunyuanWorld-1.0 published.
*   **July 26, 2025:**  HunyuanWorld-1.0, the first open-source, simulation-capable, immersive 3D world generation model, is released!

---

## 🌐 **Community & Resources**

*   **Official Site:** [![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage&height=22px)](https://3d.hunyuan.tencent.com/sceneTo3D)
*   **Hugging Face Models:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg?height=22px)](https://huggingface.co/tencent/HunyuanWorld-1)
*   **Page:** [![Page](https://img.shields.io/badge/Page-bb8a2e.svg?logo=github&height=22px)](https://3d-models.hunyuan.tencent.com/world/)
*   **Technical Report:** [![Report](https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv&height=22px)](https://arxiv.org/abs/2507.21809)
*   **Discord:** [![Discord](https://img.shields.io/badge/Discord-white.svg?logo=discord&height=22px)](https://discord.gg/dNBrdrGGMa)
*   **X (Twitter):** [![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x&height=22px)](https://x.com/TencentHunyuan)

---

## ⚙️ **HunyuanWorld 1.0: Technical Overview**

### **Abstract**

HunyuanWorld 1.0 introduces a novel framework for generating immersive, explorable, and interactive 3D worlds from text or images. It bridges the gap between video-based and 3D-based approaches by combining the strengths of both. The model uses panoramic world proxies to create 360° experiences, offering seamless integration with existing graphics pipelines and facilitating augmented interactivity with disentangled object representations.

### **Architecture**

HunyuanWorld-1.0 uses panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to create high-quality, scene-scale 360° 3D worlds from text and image inputs.

<img src="assets/arch.jpg" alt="HunyuanWorld 1.0 Architecture" width="600">

### **Performance**

HunyuanWorld 1.0 demonstrates state-of-the-art performance across several benchmark tests:

**Text-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| ...              | ...        | ...     | ...        | ...       |
| HunyuanWorld 1.0 | **40.8**   | **5.8** | **4.4**    | **24.3**  |

**Image-to-Panorama Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| ...              | ...        | ...     | ...        | ...       |
| HunyuanWorld 1.0 | **45.2**   | **5.8** | **4.3**    | **85.1**  |

**Text-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| ...              | ...        | ...     | ...        | ...       |
| HunyuanWorld 1.0 | **34.6**   | **4.3** | **4.2**    | **24.0**  |

**Image-to-World Generation:**

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | ---------- | ------- | ---------- | --------- |
| ...              | ...        | ...     | ...        | ...       |
| HunyuanWorld 1.0 | **36.2**   | **4.6** | **3.9**    | **84.5**  |

### **Visual Results**

Experience the immersive 3D worlds generated by HunyuanWorld 1.0:

<img src="assets/panorama1.gif" alt="Panorama Example 1" width="500">
<img src="assets/panorama2.gif" alt="Panorama Example 2" width="500">
<img src="assets/roaming_world.gif" alt="Roaming World Example" width="500">

---

## 📦 **Model Zoo**

Explore the available pre-trained models:

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
| -------------------------------- | ----------------------------- | ---------- | ----- | ---------------------------------------------------------------------------------------------------- |
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

---

## 🚀 **Get Started**

### **1. Environment Setup**

Ensure you have the necessary dependencies:

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml
# Install Real-ESRGAN, ZIM, and Draco. Follow the instructions in the original README.
# ...
```

### **2. Code Usage**

**Image to World Generation:**

```python
# First, generate a Panorama image with An Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Second, using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2
# And then you get your WORLD SCENE!!
```

**Text to World Generation:**

```python
# First, generate a Panorama image with A Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Second, using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# And then you get your WORLD SCENE!!
```

### **3. Quantization and Cache**
To optimize the memory and speed:

```python
# Image to World generation
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache

# Text to World generation
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### **4. Quick Start**

Run the `test.sh` script for a quick example:

```bash
bash scripts/test.sh
```

### **5. 3D World Viewer**

Visualize your generated 3D worlds using the ModelViewer tool:

1.  Open `modelviewer.html` in your browser.
2.  Upload the generated 3D scene files.
3.  Explore your real-time playable 3D experience!

<img src="assets/quick_look.gif" alt="Quick Look at 3D Viewer" width="500">

---

## 📝 **Open-Source Plan**

*   \[x] Inference Code
*   \[x] Model Checkpoints
*   \[x] Technical Report
*   \[x] Lite Version
*   \[ ] Voyager (RGBD Video Diffusion)

---

## 📖 **Citation**

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

## 📧 **Contact**

For any questions, please reach out to tengfeiwang12@gmail.com.

---

## 🙏 **Acknowledgements**

Thank you to the contributors of Stable Diffusion, FLUX, diffusers, HuggingFace, Real-ESRGAN, ZIM, GroundingDINO, MoGe, Worldsheet, and WorldGen.