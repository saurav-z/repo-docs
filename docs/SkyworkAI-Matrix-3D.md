# Matrix-3D: Generate Interactive 3D Worlds from Text and Images 

Matrix-3D revolutionizes 3D world generation by enabling users to create and explore expansive, omnidirectional 3D environments from simple text prompts or images. [Explore the Matrix-3D Repository](https://github.com/SkyworkAI/Matrix-3D) for immersive world generation!

[![Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
![Badge](https://img.shields.io/badge/version-v0.1.0-green)
[![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)

**Key Features:**

*   **Omnidirectional Exploration:** Generate 360-degree explorable 3D scenes.
*   **Text and Image Input:**  Create scenes from text prompts or images.
*   **Customizable Trajectories:** Define your own camera movements.
*   **High-Quality Output:** Produce diverse and detailed 3D scenes.
*   **Speed-Quality Balance:** Choose from rapid or detailed 3D reconstruction methods.

##  How Matrix-3D Works

Matrix-3D leverages panoramic representation for wide-coverage, omnidirectional 3D world generation. It combines conditional video generation with panoramic 3D reconstruction, allowing for creation of immersive, explorable environments.  Here are some examples of what you can create!

### Image-to-Scene Generation
| Image  | Panoramic Video | 3D Scene |
|---|---|---|
|  <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_2.gif"  height="150" width="300">  | <img src="asset/i2p/i2p_2_3D.gif" height="150">  |
|  <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"> |  <img src="asset/i2p/i2p_1.gif"  height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150">  |

### Text-to-Scene Generation
| Text  | Panoramic Video | 3D Scene |
|---|---|---|
|  A floating island with a waterfall  | <img src="asset/t2p/t2p_1.gif"  height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150">  |
|  an impressionistic winter landscape | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" > | <img src="asset/t2p/t2p_2_3D.gif" height="150">  |

## Getting Started

### Installation

1.  **Clone the repository:**

    ```bash
    git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
    cd Matrix-3D
    ```
2.  **Create and activate a Conda environment:**

    ```bash
    conda create -n matrix3d python=3.10
    conda activate matrix3d
    ```
3.  **Install dependencies (with CUDA 12.4 support):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```
4.  **Run the installation script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

### Pretrained Models

Download the necessary pretrained models from Hugging Face:

*   [Text2PanoImage](https://huggingface.co/Skywork/Matrix-3D)
*   [PanoVideoGen-480p](https://huggingface.co/Skywork/Matrix-3D)
*   [PanoVideoGen-720p](https://huggingface.co/Skywork/Matrix-3D)
*   [PanoLRM-480p](https://huggingface.co/Skywork/Matrix-3D)

### Usage

1.  **Download Checkpoints:**

    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

3.  **Step-by-Step Generation:**

    *   **Step 1: Generate Panorama Image (Text-to-Panorama or Image-to-Panorama)**

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

    *   **Step 2: Generate Panoramic Video**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```

        For low VRAM, run the video generation step with VRAM management:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --enable_vram_management
        ```

    *   **Step 3: Extract 3D Scene**

        *   Optimization-based reconstruction:

            ```bash
            python code/panoramic_video_to_3DScene.py \
                --inout_dir="./output/example1" \
                --resolution=720
            ```
        *   Feed-forward reconstruction:

            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
            ```

##  Create Your Own Scenes with Custom Camera Trajectories

| Movement Mode | Trajectory | Panoramic Video | 3D Scene |
|---|---|---|---|
| S-curve Travel | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

Choose from `Straight Travel`, `S-curve Travel`, or `Forward on the Right` using the `--movement_mode` flag in `code/panoramic_image_to_video.py`. Alternatively, use your own camera trajectory in `.json` format.

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

## Gradio Demo

Explore the Matrix-3D workflow with our interactive Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

**GPU Configuration Notes:**

*   **Single GPU:** Supports text-video-3D generation (requires at least 62GB of memory).
*   **Multiple GPUs:** Supports both text-video-3D and image-video-3D generation.

## Citation

If you find Matrix-3D useful, please cite our work:

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}
```

## Acknowledgements

This project builds on the work of several other projects:

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

For questions or feature requests, please open an issue on our GitHub repository!