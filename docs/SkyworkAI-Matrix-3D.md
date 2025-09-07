<h1 align="center">
    Matrix-3D: Generate and Explore Omnidirectional 3D Worlds
</h1>

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>  
</div>

<div align="center">
    <a href="https://github.com/SkyworkAI/Matrix-3D">
        <img src="https://img.shields.io/badge/GitHub-SkyworkAI/Matrix--3D-blue?style=flat&logo=github" alt="GitHub">
    </a>
    <a href="https://matrix-3d.github.io/">
        <img src="https://img.shields.io/badge/Project_Page-orange?style=flat" alt="Project Page">
    </a>
    <a href="https://huggingface.co/Skywork/Matrix-3D">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face">
    </a>
    <a href="https://arxiv.org/pdf/2508.08086">
        <img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv">
    </a>
    <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version">
</div>

## Introduction

**Unleash your creativity and build immersive 3D worlds with Matrix-3D, an innovative approach to omnidirectional 3D world generation.** This project leverages panoramic representations, combining conditional video generation and 3D reconstruction to create explorable environments.

**Key Features:**

*   **Expansive Scene Generation:** Generate vast, 360-degree explorable scenes beyond traditional limitations.
*   **High Controllability:** Easily generate worlds using text prompts or images, with customizable trajectories and infinite scene possibilities.
*   **Strong Generalization:** Leveraging self-developed 3D data and video model priors to generate diverse, high-quality scenes.
*   **Speed-Quality Balance:** Choose between rapid and detailed 3D reconstruction methods.

## What's New
*   **[Sep 02, 2025]:** 🚀 5B model with low-VRAM mode, requiring only 12G VRAM!
*   **[Aug 29, 2025]:** 🚀 Interactive [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) available!
*   **[Aug 25, 2025]:** 🚀 Script to run generation with 19G VRAM!
*   **[Aug 12, 2025]:** 🚀 Code, technical report, and project page released!

## Image-to-Scene Generation

| Image                     | Panoramic Video           | 3D Scene                   |
| ------------------------- | ------------------------- | -------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_2.gif"  height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150"> |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_1.gif"  height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150"> |

## Text-to-Scene Generation

| Text                                                                | Panoramic Video           | 3D Scene                   |
| ------------------------------------------------------------------- | ------------------------- | -------------------------- |
| A floating island with a waterfall   | <img src="asset/t2p/t2p_1.gif"  height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| an impressionistic winter landscape | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" > | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

**Related Project:** For Real-Time Interactive Long-Sequence World Models, check out [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux with NVIDIA GPU.

1.  **Clone the repository:**

    ```bash
    git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
    cd Matrix-3D
    ```

2.  **Create a Conda environment:**

    ```bash
    conda create -n matrix3d python=3.10
    conda activate matrix3d
    ```

3.  **Install PyTorch and dependencies (CUDA 12.4 recommended):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name           | Description                         | Download                                                                           |
| -------------------- | ----------------------------------- | ---------------------------------------------------------------------------------- |
| Text2PanoImage       |  -                                  | [Link](https://huggingface.co/Skywork/Matrix-3D)                                   |
| PanoVideoGen-480p    |  -                                  | [Link](https://huggingface.co/Skywork/Matrix-3D)                                   |
| PanoVideoGen-720p    |  -                                  | [Link](https://huggingface.co/Skywork/Matrix-3D)                                   |
| PanoVideoGen-720p-5B  |  -                                  | [Link](https://huggingface.co/Skywork/Matrix-3D)                                   |
| PanoLRM-480p         |  -                                  | [Link](https://huggingface.co/Skywork/Matrix-3D)                                   |

### GPU VRAM Requirements

| Model Name           | VRAM (Approximate) | VRAM with Low-VRAM Mode |
| -------------------- | ------------------ | ----------------------- |
| Text2PanoImage       | \~16 GB            | -                       |
| PanoVideoGen-480p    | \~40 GB            | \~15 GB                 |
| PanoVideoGen-720p    | \~60 GB            | \~19 GB                 |
| PanoVideoGen-720p-5B  | \~19 GB            | \~12 GB                 |
| PanoLRM-480p         | \~80 GB            | -                       |

**Note:**  PanoLRM inference uses significant VRAM; the optimization-based reconstruction (described below) is an alternative, using ~10 GB.

## Usage

*   **Checkpoint Download:**
    ```bash
    python code/download_checkpoints.py
    ```
*   **One-Command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

*   **Step-by-Step Generation:**

    1.  **Text/Image to Panorama Image:**

        *   From Text:
            ```bash
            python code/panoramic_image_generation.py \
                --mode=t2p \
                --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
                --output_path="./output/example1"
            ```

        *   From Image:
            ```bash
            python code/panoramic_image_generation.py \
                --mode=i2p \
                --input_image_path="./data/image1.jpg" \
                --output_path="./output/example1"
            ```
            Ensure the prompt is in `output/example1/prompt.txt` for step 2.

    2.  **Generate Panoramic Video:**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```
        Choose resolution 480 or 720.

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

    3.  **Extract 3D Scene:**

        *   Optimization-Based Reconstruction:
            ```bash
             python code/panoramic_video_to_3DScene.py \
                --inout_dir="./output/example1" \
                --resolution=720
            ```
        *   Feed-Forward Reconstruction:
            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
            ```
            Replace video and pose paths.

## Create Your Own

| Movement Mode        | Trajectory                   | Panoramic Video           | 3D Scene                   |
| -------------------- | ---------------------------- | ------------------------- | -------------------------- |
| S-curve Travel       | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

*   **Movement Modes:** `Straight Travel`, `S-curve Travel`, and `Forward on the Right` (configure with `--movement_mode`).
*   **Custom Trajectories:** Use a .json file (world-to-camera matrices in OpenCV format) and the `--json_path` argument.

## Gradio Demo

Explore the Matrix-3D workflow with our interactive Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

*   **Single GPU:**  Supports text-video-3D generation (62GB+ VRAM).
*   **Multiple GPUs:**  Supports text-video-3D and image-video-3D workflows; configure GPU allocation as needed.

## Citation

If you use this project, please cite it:

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}
```

---

## Acknowledgements

This project is built upon the following:
*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

For questions or feature requests, please open an issue.