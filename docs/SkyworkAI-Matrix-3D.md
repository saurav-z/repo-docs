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

**Matrix-3D empowers you to create explorable 3D worlds from text prompts or images, offering unprecedented control and immersion.**  Explore the original repo on [GitHub](https://github.com/SkyworkAI/Matrix-3D).

## Key Features

*   **Omnidirectional Exploration:** Generate 360-degree navigable 3D scenes.
*   **Text-to-Scene & Image-to-Scene:**  Transform text descriptions or images into interactive 3D environments.
*   **High Controllability:** Customize scene trajectories and enjoy infinite extensibility with text and image inputs.
*   **Strong Generalization:** Leverage self-developed 3D data and video model priors for diverse and high-quality scene generation.
*   **Speed-Quality Balance:** Choose between rapid and detailed 3D reconstruction methods.
*   **Low VRAM Mode:** Run video generation with as little as 12GB VRAM.

## What's New

*   **September 2, 2025:** 🎉 Released a 5B model with low-VRAM mode (12GB VRAM required).
*   **August 29, 2025:** 🎉 Launched a Gradio demo!
*   **August 25, 2025:** 🎉 Provided a script for 19G VRAM generation.
*   **August 12, 2025:** 🎉 Released code, technical report, and project page.

## Image-to-Scene Generation

| Image                                                                                                                         | Panoramic Video                                                                 | 3D Scene                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br> | <img src="asset/i2p/i2p_2.gif"  height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150"> |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br> | <img src="asset/i2p/i2p_1.gif"  height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150"> |

## Text-to-Scene Generation

| Text                                                                                                                                                            | Panoramic Video                                                                 | 3D Scene                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall | <img src="asset/t2p/t2p_1.gif"  height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" > | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

**Related Project**: Explore Real-Time Interactive Long-Sequence World Models in [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux with NVIDIA GPU.

```bash
# Clone the repository
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D

# Create a new conda environment
conda create -n matrix3d python=3.10
conda activate matrix3d

# Install torch and torchvision (with GPU support, we use CUDA 12.4 Version)
pip install torch==2.7.0 torchvision==0.22.0

# Run installation script
chmod +x install.sh
./install.sh
```

## Pretrained Models

| Model Name           | Description                | Download                                               |
| -------------------- | -------------------------- | ------------------------------------------------------ |
| Text2PanoImage       | -                          | [Link](https://huggingface.co/Skywork/Matrix-3D)       |
| PanoVideoGen-480p    | -                          | [Link](https://huggingface.co/Skywork/Matrix-3D)       |
| PanoVideoGen-720p    | -                          | [Link](https://huggingface.co/Skywork/Matrix-3D)       |
| PanoVideoGen-720p-5B | -                          | [Link](https://huggingface.co/Skywork/Matrix-3D)       |
| PanoLRM-480p         | -                          | [Link](https://huggingface.co/Skywork/Matrix-3D)       |

## GPU VRAM Requirements

| Model Name          | VRAM        | VRAM with low-vram mode |
| ------------------- | ----------- | ----------------------- |
| Text2PanoImage      | ~16GB       | -                       |
| PanoVideoGen-480p   | ~40GB       | ~15GB                   |
| PanoVideoGen-720p   | ~60GB       | ~19GB                   |
| PanoVideoGen-720p-5B | ~19GB       | ~12GB                   |
| PanoLRM-480p        | ~80GB       | -                       |

**Note:** PanoLRM inference requires significant VRAM, but you can replace it with optimization-based reconstruction (about 10GB VRAM).

## Usage

*   **Checkpoint Download**

```bash
python code/download_checkpoints.py
```

*   **One-command 3D World Generation**

```bash
./generate.sh
```

*   **Step-by-Step Generation**

1.  **Text/Image to Panorama Image:**

    *   From text:

    ```bash
    python code/panoramic_image_generation.py \
        --mode=t2p \
        --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
        --output_path="./output/example1"
    ```

    *   From an image:

    ```bash
    python code/panoramic_image_generation.py \
        --mode=i2p \
        --input_image_path="./data/image1.jpg" \
        --output_path="./output/example1"
    ```

    Organize your panorama image with a prompt:
    ```
    ./output/example1
    └─ pano_img.jpg
    └─ prompt.txt
    ```

2.  **Generate Panoramic Video:**

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
  --enable_vram_management # enable this to allow model to run on devices with 19G vram.
```

*   **5B model:**
```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720 \
  --use_5b_model # enable this to generate video with light-weight 5B model.
```


3.  **Extract 3D Scene:**

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

## Create Your Own

| Movement Mode         | Trajectory                    | Panoramic Video                                    | 3D Scene                                           |
| --------------------- | ----------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| S-curve Travel        | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

You can configure movement mode and also supply custom camera trajectories.

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

## Gradio Demo

Launch the demo:

```bash
python code/matrix.py --max_gpus=1
```
### GPU Configuration:
- Single GPU (--max_gpus=1): Supports text-video-3D generation workflow. Requires 62 GB of memory.
- Multiple GPUs (--max_gpus=N, N≥2): Supports both text-video-3D and image-video-3D generation workflows. Allocate GPUs based on hardware resources.

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

This project leverages the following resources (cite them if used):

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

For questions or feature requests, please submit an issue.