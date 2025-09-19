<h1 align="center">
  Matrix-3D: Generate and Explore Omnidirectional 3D Worlds
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

**Tired of static scenes? Matrix-3D unlocks the ability to generate and explore dynamic, 360-degree explorable 3D worlds from text or images.**  [See the original repo](https://github.com/SkyworkAI/Matrix-3D).

## Key Features

*   🌎 **Omnidirectional Exploration:** Generate expansive, 360-degree explorable 3D scenes.
*   ✍️ **Text & Image Input:** Control scene generation with both text prompts and image inputs.
*   ✨ **High-Quality & Diverse Scenes:** Leverage self-developed 3D data and video model priors for rich 3D scene generation.
*   🚀 **Speed-Quality Balance:** Choose between rapid or detailed 3D reconstruction methods.
*   🚀 **Low VRAM Mode:** Generate 720p video on devices with as little as 19 GB of VRAM.
*   🚀 **5B Model:** Generate high-quality videos with fast generation and low VRAM requirements using the 5B model.

## Image-to-Scene Generation

| Image | Panoramic Video | 3D Scene |
| ----- | --------------- | -------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_2.gif"  height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150"> |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_1.gif"  height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150"> |

## Text-to-Scene Generation

| Text | Panoramic Video | 3D Scene |
| ----- | --------------- | -------- |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif"  height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</div> | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" > | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

## News
- Sep 02, 2025: 🎉 We provide a 5B model with low-VRAM mode which only requires 12G VRAM!
- Aug 29, 2025: 🎉 We provide a [gradio demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) for Matrix-3D!
- Aug 25, 2025: 🎉 We provide a  [script](#lowvram) for running the generation process with 19G VRAM!
- Aug 12, 2025: 🎉 We release the code, technical report and project page of Matrix-3D!

## Installation

**Prerequisites:** Linux system with NVIDIA GPU.  We recommend using CUDA 12.4

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

3.  **Install PyTorch (with GPU support):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```

4.  **Run the Installation Script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name           | Description                                   | Download                                                                   |
| :------------------- | :-------------------------------------------- | :------------------------------------------------------------------------- |
| Text2PanoImage       | Text-to-Panorama Image Generation           | [Link](https://huggingface.co/Skywork/Matrix-3D)                                         |
| PanoVideoGen-480p    | Panorama Video Generation (480p)              | [Link](https://huggingface.co/Skywork/Matrix-3D)                                         |
| PanoVideoGen-720p    | Panorama Video Generation (720p)              | [Link](https://huggingface.co/Skywork/Matrix-3D)                                         |
| PanoVideoGen-720p-5B | Panorama Video Generation (720p, 5B model)    | [Link](https://huggingface.co/Skywork/Matrix-3D)                                         |
| PanoLRM-480p         | Panorama Low Resolution Model 3D Reconstruction (480p) | [Link](https://huggingface.co/Skywork/Matrix-3D)          |

## GPU VRAM Requirements

| Model Name           | VRAM (approximate) | VRAM with low-vram mode |
| :------------------- | :----------------- | :----------------------- |
| Text2PanoImage       | \~16GB            | -                        |
| PanoVideoGen-480p    | \~40GB            | \~15GB                  |
| PanoVideoGen-720p    | \~60GB            | \~19GB                  |
| PanoVideoGen-720p-5B | \~19GB            | \~12GB                   |
| PanoLRM-480p         | \~80GB            | -                        |

**Note:** The inference of PanoLRM will take lots of VRAM, but it is optional, you can replace it with the optimization-based reconstruction (see below), which only takes about 10GB VRAM.

## Usage

1.  **Download Checkpoints:**

    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

    Or, follow the steps below:

3.  **Step 1: Text/Image to Panorama Image:**

    *   From text:

        ```bash
        python code/panoramic_image_generation.py \
            --mode=t2p \
            --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
            --output_path="./output/example1"
        ```

    *   From image:

        ```bash
        python code/panoramic_image_generation.py \
            --mode=i2p \
            --input_image_path="./data/image1.jpg" \
            --output_path="./output/example1"
        ```

4.  **Step 2: Generate Panoramic Video:**

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
    ```

    *   **Low VRAM Mode:** For lower VRAM GPUs:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --enable_vram_management
        ```

    *   **5B Model:** For faster generation and lower VRAM usage:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --use_5b_model
        ```

5.  **Step 3: Extract 3D Scene:**

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

| Movement Mode | Trajectory | Panoramic Video | 3D Scene |
| :------------ | :--------- | --------------- | -------- |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>S-curve Travel</div> | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>Forward on the Right</div> | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

*   Movement Modes: `Straight Travel`, `S-curve Travel`, `Forward on the Right`. Configure with `--movement_mode` in `code/panoramic_image_to_video.py`.
*   Custom Trajectories:  Provide your own camera trajectory in `.json` format.

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720 \
      --json_path YOUR_TRAJECTORY_FILE.json
    ```

    Use `code/generate_example_camera.py` to create custom camera trajectories.  Camera matrices are world-to-camera matrices in OpenCV format.

## Gradio Demo

Launch the Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

*   **GPU Configuration:**
    *   Single GPU (`--max_gpus=1`): Text-video-3D generation workflow.  Requires at least 62 GB memory.
    *   Multiple GPUs (`--max_gpus=N, N≥2`): Supports text-video-3D and image-video-3D generation workflows.

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

This project builds upon the following:

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

For questions or feature requests, please [post an issue](https://github.com/SkyworkAI/Matrix-3D/issues).