<h1 align="center">
    Matrix-3D: Generate and Explore Omnidirectional 3D Worlds
</h1>
<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>
<div align="center">
  <a href="https://github.com/SkyworkAI/Matrix-3D">
    <img src="https://img.shields.io/badge/View_on_GitHub-SkyworkAI/Matrix--3D-blue?logo=github" alt="GitHub">
  </a>
  <a href="https://matrix-3d.github.io/">
    <img src="https://img.shields.io/badge/Project_Page-orange" alt="Project Page">
  </a>
  <a href="https://huggingface.co/Skywork/Matrix-3D">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model">
  </a>
  <a href="https://arxiv.org/pdf/2508.08086">
    <img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Technical Report">
  </a>
  <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version">
</div>

**Matrix-3D empowers you to create and explore immersive, 360-degree explorable 3D worlds from text or images.**

## Key Features

*   **Expansive 3D Scene Generation:** Create large-scale, fully explorable 3D environments, unlike traditional scene generation approaches.
*   **High Controllability:**  Generate scenes from text or images with customizable trajectories, offering infinite extensibility.
*   **Robust Generalization:** Built upon our self-developed 3D data and video model priors, enabling diverse and high-quality 3D scene creation.
*   **Speed-Quality Balance:** Choose from two panoramic 3D reconstruction methods for rapid or detailed scene generation.

## What's New
*   **[September 2, 2025]:**  5B model released, requiring only 12GB of VRAM!
*   **[August 29, 2025]:** Interactive Gradio demo available.
*   **[August 25, 2025]:** Script for 19GB VRAM generation.
*   **[August 12, 2025]:** Code, technical report, and project page release.

## Examples: Image-to-Scene & Text-to-Scene

(Image/Text input with corresponding Panoramic Video & 3D Scene outputs are displayed in tables. Refer to the original README for the complete visual examples)

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
    cd Matrix-3D
    ```
2.  **Create a Conda Environment:**
    ```bash
    conda create -n matrix3d python=3.10
    conda activate matrix3d
    ```
3.  **Install Dependencies (with GPU support):**
    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name              | Description                   | Download                                                 |
| :---------------------- | :---------------------------- | :------------------------------------------------------- |
| Text2PanoImage          | Text-to-Panorama Image Generation | [Link](https://huggingface.co/Skywork/Matrix-3D)        |
| PanoVideoGen-480p       | 480p Video Generation         | [Link](https://huggingface.co/Skywork/Matrix-3D)        |
| PanoVideoGen-720p       | 720p Video Generation         | [Link](https://huggingface.co/Skywork/Matrix-3D)        |
| PanoVideoGen-720p-5B    | 720p (5B model) | [Link](https://huggingface.co/Skywork/Matrix-3D)        |
| PanoLRM-480p            | Panoramic LRM Model (480p)     | [Link](https://huggingface.co/Skywork/Matrix-3D)        |

## GPU VRAM Requirements

| Model Name              | VRAM     | VRAM with Low-VRAM Mode |
| :---------------------- | :------- | :----------------------- |
| Text2PanoImage          | ~16GB    | -                        |
| PanoVideoGen-480p       | ~40GB    | ~15GB                    |
| PanoVideoGen-720p       | ~60GB    | ~19GB                    |
| PanoVideoGen-720p-5B    | ~19GB    | ~12GB                    |
| PanoLRM-480p            | ~80GB    | -                        |

## Usage

1.  **Download Checkpoints:**
    ```bash
    python code/download_checkpoints.py
    ```
2.  **One-Command 3D World Generation:**
    ```bash
    ./generate.sh
    ```

3.  **Step-by-Step Generation:**

    *   **Step 1: Panorama Image Generation** (from Text or Image - examples provided)
    *   **Step 2: Panoramic Video Generation:** (examples provided)
         * Low VRAM Mode (for 720p generation with 19GB VRAM):
             ```bash
             VISIBLE_GPU_NUM=1
             torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
               --inout_dir="./output/example1"  \
               --resolution=720 \
               --enable_vram_management
             ```
         * 5B Model (for fast generation with lower VRAM)
             ```bash
             VISIBLE_GPU_NUM=1
             torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
               --inout_dir="./output/example1"  \
               --resolution=720 \
               --use_5b_model
             ```

    *   **Step 3: 3D Scene Extraction** (Optimization-based or feed-forward - examples provided)

## Create Your Own

(Examples of custom movement modes and camera trajectory usage, refer to the original README for the complete visual examples)

## Gradio Demo

Run the following to launch the Gradio demo:
```bash
python code/matrix.py --max_gpus=1
```
*  Single GPU (`--max_gpus=1`): Text-video-3D generation (requires at least 62GB of memory).
*   Multiple GPUs (`--max_gpus=N, N≥2`): Supports both workflows; adjust GPU allocation accordingly.

## Citation

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}
```

## Acknowledgements

(List of Acknowledgements)

## Contact

Feel free to open an issue for questions or feature requests.