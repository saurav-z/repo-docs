<h1 align="center">
    Matrix-3D: Generate and Explore 3D Worlds from Text or Images
</h1>

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">
  <a href="https://matrix-3d.github.io/"><img src="https://img.shields.io/badge/📄-Project_Page-orange" alt="Project Page"></a>
  <a href="https://huggingface.co/Skywork/Matrix-3D"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model"></a>
  <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version">
  <a href="https://arxiv.org/pdf/2508.08086"><img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Technical Report"></a>
</div>

## Overview

**Matrix-3D revolutionizes 3D world creation, enabling the generation of expansive, explorable scenes from simple text prompts or images.** Built upon state-of-the-art video generation and 3D reconstruction techniques, this project offers unprecedented control, broad scene generation capabilities, and a speed-quality balance. Explore the future of immersive experiences with Matrix-3D! [See the original repo](https://github.com/SkyworkAI/Matrix-3D).

**Key Features:**

*   ✨ **Omnidirectional Exploration:** Generate and explore 360-degree scenes.
*   🖼️ **Text-to-Scene and Image-to-Scene:** Create 3D worlds from text descriptions or images.
*   🚀 **High Controllability:** Customize trajectories and scene details.
*   🌍 **Strong Generalization:** Generate diverse, high-quality scenes thanks to self-developed 3D and video model priors.
*   ⚡️ **Speed-Quality Balance:** Choose between fast and detailed 3D reconstruction methods.

## ✨ News

*   **September 02, 2025:** 🎉 5B Model Released! Achieve high-quality results with only 12GB VRAM!
*   **August 29, 2025:** 🎉 Interactive Gradio Demo Available! Explore Matrix-3D through our new demo.
*   **August 25, 2025:** 🎉 Low-VRAM Script Released! Generate 720p videos with just 19GB VRAM.
*   **August 12, 2025:** 🎉 Code, Technical Report, and Project Page Launch!

## 🖼️ Examples: Image-to-Scene Generation

| Image                                                                                                                                                                 | Panoramic Video                                                           | 3D Scene                                                                                                            |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------ |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br>                                                                                        | <img src="asset/i2p/i2p_2.gif" height="150" width="300">                   | <img src="asset/i2p/i2p_2_3D.gif" height="150">                                                                    |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br>                                                                                        | <img src="asset/i2p/i2p_1.gif" height="150" width="300">                   | <img src="asset/i2p/i2p_1_3D.gif" height="150">                                                                    |

## 📝 Examples: Text-to-Scene Generation

| Text                                                                                                                                                                                                                                                                                         | Panoramic Video                                                            | 3D Scene                                                                                                            |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| <div style="background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall | <img src="asset/t2p/t2p_1.gif" height="150" width="300">                  | <img src="asset/t2p/t2p_1_3D.gif" height="150">                                                                    |
| <div style="background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape | <img src="asset/t2p/t2p_2.gif" height="150" width="300">                  | <img src="asset/t2p/t2p_2_3D.gif" height="150">                                                                    |

## 📦 Installation

**Prerequisites:** Linux with NVIDIA GPU, CUDA 12.4 (recommended).

```bash
# Clone the repository
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D

# Create a new conda environment
conda create -n matrix3d python=3.10
conda activate matrix3d

# Install torch and torchvision (with GPU support, we use CUDA 12.4 Version)
pip install torch==2.7.0 torchvision==0.22.0

#Run installation script
chmod +x install.sh
./install.sh
```

## 💾 Pretrained Models

| Model Name              | Description                                       | Download                                                                     |
| :---------------------: | :------------------------------------------------ | :---------------------------------------------------------------------------: |
| Text2PanoImage          | Generate panorama image from text prompts        |  [Link](https://huggingface.co/Skywork/Matrix-3D)                           |
| PanoVideoGen-480p       | Panorama Video Generation (480p)                 |  [Link](https://huggingface.co/Skywork/Matrix-3D)                           |
| PanoVideoGen-720p       | Panorama Video Generation (720p)                 |  [Link](https://huggingface.co/Skywork/Matrix-3D)                           |
| PanoVideoGen-720p-5B    | Panorama Video Generation (720p, 5B model)        |  [Link](https://huggingface.co/Skywork/Matrix-3D)                           |
| PanoLRM-480p            | Efficient 3D Reconstruction                       |  [Link](https://huggingface.co/Skywork/Matrix-3D)                           |

## 📊 GPU VRAM Requirements

*   **Minimum Requirement:** 16GB VRAM to run the entire pipeline.

| Model Name           | VRAM (Approx.) | VRAM with Low-VRAM Mode |
| :------------------: | :-------------: | :---------------------: |
| Text2PanoImage       |      ~16GB      |           -             |
| PanoVideoGen-480p    |      ~40GB      |          ~15GB          |
| PanoVideoGen-720p    |      ~60GB      |          ~19GB          |
| PanoVideoGen-720p-5B |      ~19GB      |          ~12GB          |
| PanoLRM-480p         |      ~80GB      |           -             |

**Note:** PanoLRM's inference requires significant VRAM, but can be replaced with optimization-based reconstruction (approx. 10GB VRAM).

## 🎮 Usage

*   **1.  Download Checkpoints:**
    ```bash
    python code/download_checkpoints.py
    ```
*   **2.  One-Command 3D World Generation:**
    ```bash
    ./generate.sh
    ```

*   **3. Step-by-Step Generation:**

    *   **Step 1: Text/Image to Panorama Image**

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

    *   **Step 3: Extract 3D Scene**

        *   Optimization-based Reconstruction:

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

## 🎬 Create Your Own: Movement Modes & Trajectories

| Movement Mode        | Trajectory                                         | Panoramic Video                         | 3D Scene                                 |
| :------------------- | :------------------------------------------------- | :--------------------------------------- | :--------------------------------------- |
| S-curve Travel       | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

You can customize the camera movement with `--movement_mode` in `code/panoramic_image_to_video.py`, or provide your own `.json` trajectory:

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```
Use `code/generate_example_camera.py` to create your own camera trajectory.

## 🖱️ Gradio Demo

Explore Matrix-3D interactively:

```bash
python code/matrix.py --max_gpus=1
```
*   **GPU Configuration Notes:**

    *   Single GPU (`--max_gpus=1`): Supports text-video-3D generation (requires ~62GB VRAM).
    *   Multiple GPUs (`--max_gpus=N, N≥2`): Supports both text-video-3D and image-video-3D.

## 📚 Citation

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}
```

## 🤝 Acknowledgements

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## 📧 Contact

For questions or feature requests, please open an issue.