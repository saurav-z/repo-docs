# Matrix-3D: Generate and Explore Omnidirectional 3D Worlds

Matrix-3D revolutionizes 3D world generation by allowing you to create and explore expansive, 360-degree environments from text or images.

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">
  <a href="https://matrix-3d.github.io/"><img src="https://img.shields.io/badge/📄-Project_Page-orange" alt="Project Page"/></a>
  <a href="https://huggingface.co/Skywork/Matrix-3D"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model"/></a>
  <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version"/>
  <a href="https://arxiv.org/pdf/2508.08086"><img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Technical Report"/></a>
</div>

## Key Features

*   **Omnidirectional Exploration:** Generate scenes offering complete 360-degree exploration.
*   **Text & Image Input:**  Create worlds from both textual descriptions and images.
*   **Customizable Trajectories:**  Define the path and movement within your generated worlds.
*   **High-Quality Output:**  Leverages advanced 3D data and video model priors for diverse and high-quality scenes.
*   **Speed-Quality Balance:** Choose between rapid and detailed 3D reconstruction methods.
*   **Low VRAM Option:** Generate 720p videos on 19GB VRAM, and 5B models.

## What's New

*   **[September 02, 2025]** - 🎉 5B model with low-VRAM mode, requiring only 12G VRAM!
*   **[August 29, 2025]** - 🎉 Gradio Demo Available! See it in action [here](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo).
*   **[August 25, 2025]** - 🎉 Script released to run generation with 19G VRAM.
*   **[August 12, 2025]** - 🎉 Code, technical report, and project page released!

## Examples: Image-to-Scene and Text-to-Scene Generation

**(See the original README for example tables and images)**

## Installation

**Prerequisites:**  Tested on Linux systems with NVIDIA GPUs.

**Steps:**

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
3.  **Install Dependencies (with CUDA 12.4 support):**
    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```
4.  **Run the Installation Script:**
    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

**(See the original README for a table of model names, descriptions, and download links)**

## GPU VRAM Requirements

**(See the original README for a table of model names and VRAM usages)**

**Important Notes:**

*   The optimization-based 3D reconstruction can be replaced by the feed-forward reconstruction for low VRAM usage.

## Usage

1.  **Checkpoint Download:**
    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation:**
    ```bash
    ./generate.sh
    ```

3.  **Step-by-Step Generation:**

    *   **Step 1: Text/Image to Panorama Image:**
        *   **From Text:**
            ```bash
            python code/panoramic_image_generation.py \
                --mode=t2p \
                --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
                --output_path="./output/example1"
            ```
        *   **From Image:**
            ```bash
            python code/panoramic_image_generation.py \
                --mode=i2p \
                --input_image_path="./data/image1.jpg" \
                --output_path="./output/example1"
            ```
    *   **Step 2: Generate Panoramic Video:**
        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```
        *   **Low VRAM Mode:**
            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --enable_vram_management
            ```
        *   **5B Model:**
            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --use_5b_model
            ```
    *   **Step 3: Extract 3D Scene:**
        *   **Optimization-based:**
            ```bash
             python code/panoramic_video_to_3DScene.py \
                --inout_dir="./output/example1" \
                --resolution=720
            ```
        *   **Feed-forward:**
            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
            ```

## Create Your Own Worlds

**(See the original README for tables and code examples)**

## Gradio Demo

```bash
python code/matrix.py --max_gpus=1
```

**Notes on GPU Configuration:**

*   `--max_gpus=1`: Text-video-3D generation (requires at least 62 GB).
*   `--max_gpus=N, N≥2`:  Supports both text-video-3D and image-video-3D, adjust based on hardware.

## Citation

**(See the original README for BibTex)**

## Acknowledgements

**(See the original README for a list of acknowledgements)**

## Contact

For questions or feature requests, please open an issue on our [GitHub repository](https://github.com/SkyworkAI/Matrix-3D).