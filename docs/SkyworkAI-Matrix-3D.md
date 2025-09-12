<h1 align="center">
    Matrix-3D: Generate Interactive 3D Worlds from Text or Images
</h1>
<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">

[![📄 Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
![Badge](https://img.shields.io/badge/version-v0.1.0-green)
[![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)
</div>

**Unleash your imagination and explore immersive 3D worlds created from text prompts or images with Matrix-3D!** 

[View the original repository on GitHub](https://github.com/SkyworkAI/Matrix-3D).

## Key Features

*   **Omnidirectional 3D World Generation:** Create expansive, explorable 360-degree scenes.
*   **Text and Image Input:** Generate scenes from text prompts or input images.
*   **Customizable Trajectories:** Explore scenes with personalized camera paths.
*   **High-Quality & Diverse Scenes:**  Leveraging advanced 3D data and video model priors.
*   **Speed-Quality Balance:** Choose between rapid and detailed 3D reconstruction methods.

## What's New

*   **September 2, 2025:** 🚀  5B model with low-VRAM mode (12G VRAM requirement).
*   **August 29, 2025:** 🎉 Integrated [Gradio demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) for easy visualization.
*   **August 25, 2025:** 🎉  [Script](#lowvram) provided for running generation with 19G VRAM.
*   **August 12, 2025:** 🎉 Code, technical report, and project page released!

## Image-to-Scene Generation
**(Interactive GIFs Removed for Brevity)**

## Text-to-Scene Generation
**(Interactive GIFs Removed for Brevity)**

**Related Project**: Discover [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) for Real-Time Interactive Long-Sequence World Models.

## Installation

**Prerequisites:** Linux system with an NVIDIA GPU.

1.  **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
    cd Matrix-3D
    ```

2.  **Create and Activate Conda Environment:**

    ```bash
    conda create -n matrix3d python=3.10
    conda activate matrix3d
    ```

3.  **Install PyTorch and Torchvision (with CUDA 12.4):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```

4.  **Run Installation Script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name           | Description                  | Download                                                                 |
| :------------------- | :--------------------------- | :----------------------------------------------------------------------- |
| Text2PanoImage       | Text to Panoramic Image     | [Link](https://huggingface.co/Skywork/Matrix-3D)                          |
| PanoVideoGen-480p    | Panoramic Video (480p)      | [Link](https://huggingface.co/Skywork/Matrix-3D)                          |
| PanoVideoGen-720p    | Panoramic Video (720p)      | [Link](https://huggingface.co/Skywork/Matrix-3D)                          |
| PanoVideoGen-720p-5B | Panoramic Video (720p, 5B)  | [Link](https://huggingface.co/Skywork/Matrix-3D)                          |
| PanoLRM-480p         | Panoramic LRM (480p)       | [Link](https://huggingface.co/Skywork/Matrix-3D)                          |

## GPU VRAM Requirements

| Model Name           | VRAM (approximate) | VRAM with Low-VRAM Mode |
| :------------------- | :----------------- | :---------------------- |
| Text2PanoImage       | ~16GB              | -                       |
| PanoVideoGen-480p    | ~40GB              | ~15GB                   |
| PanoVideoGen-720p    | ~60GB              | ~19GB                   |
| PanoVideoGen-720p-5B | ~19GB              | ~12GB                   |
| PanoLRM-480p         | ~80GB              | -                       |

**Note:** PanoLRM inference uses significant VRAM. Consider the optimization-based reconstruction as an alternative (~10GB VRAM).

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

    *   **Step 1: Generate Panorama Image (Text or Image):**
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

    *   **Step 2: Generate Panoramic Video:**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```

        *   **Low VRAM Mode:**  Enable with `--enable_vram_management`.
        *   **5B Model:** Enable with `--use_5b_model`.

    *   **Step 3: Extract 3D Scene:**
        *   **Optimization-Based Reconstruction:**

            ```bash
            python code/panoramic_video_to_3DScene.py \
               --inout_dir="./output/example1" \
               --resolution=720
            ```

        *   **Feed-Forward Reconstruction:**

            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
               --video_path="./data/case1/sample_video.mp4" \
               --pose_path='./data/case1/sample_cam.json' \
               --out_path='./output/example2'
            ```

## Create Your Own

**(Table Showing Movement Modes, Trajectories, and Visual Examples.  GIFs Removed for Brevity)**

*   **Movement Modes:**  `Straight Travel`, `S-curve Travel`, `Forward on the Right`. Configure with `--movement_mode`.
*   **Custom Camera Trajectories:** Use a .json file.  Example: `./data/test_cameras/test_cam_front.json`.  Use `code/generate_example_camera.py` to create your own.

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

## Gradio Demo

Run the following command:

```bash
python code/matrix.py --max_gpus=1
```

*   **Single GPU:** Supports text-video-3D workflow (62GB+ memory recommended).
*   **Multiple GPUs:** Supports both text-video-3D and image-video-3D workflows.

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

This project builds upon the following resources:

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

For any questions or feature requests, please submit an issue.
```
Key improvements and explanations:

*   **SEO Optimization:** Focused keywords ("3D world generation," "interactive 3D," "panoramic video") are integrated throughout the text.  Headings are used logically.
*   **One-Sentence Hook:**  Provides an engaging introduction.
*   **Concise and Clear:**  Information is presented more directly and efficiently.
*   **Bullet Points:**  Key features are highlighted for quick readability.
*   **Summarized Examples:**  The full image/video/3D scene tables are kept shorter (GIFs removed).
*   **Clear Instructions:**  Installation and usage steps are more structured.
*   **Low VRAM and 5B model emphasis:** The low vram and 5B models are highlighted for their importance.
*   **Emphasis on GPU Memory Needs:** Explicitly states GPU VRAM requirements, which is important for users.
*   **Clearer Command Instructions**: Commands are displayed with better formatting.
*   **Demo Instructions:** Correct and helpful info on the Gradio Demo.
*   **Conciseness:** The original README was pared down to the core information.
*   **Replaced long tables with concise summaries**: Removed unneeded tables, leaving only the key information.
*   **Focus on User Value:** Highlights the benefits for the user.
*   **Clear Call to Action:** Encourages users to explore and create.
*   **Better Structure:** Improved organization with headings and subheadings to improve readability.
*   **Removed redundancies and unnecessary details:** Streamlined the content.