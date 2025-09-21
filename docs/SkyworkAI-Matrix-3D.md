# Matrix-3D: Create Immersive, Explorable 3D Worlds from Text or Images

**Transform your imagination into interactive 3D experiences with Matrix-3D, generating expansive, explorable worlds from simple text prompts or images.  [[Original Repo](https://github.com/SkyworkAI/Matrix-3D)]**

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">
  [![📄 Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
  [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
  ![Badge](https://img.shields.io/badge/version-v0.1.0-green)
  [![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)
</div>

## Key Features

*   **Omnidirectional Exploration:** Generate complete 360-degree scenes for immersive exploration.
*   **Text and Image Input:** Create worlds from text descriptions or existing images.
*   **Customizable Experiences:** Control the camera trajectory and generate diverse scenes.
*   **High-Quality Results:** Built upon advanced video and 3D model priors to ensure high-quality scene generation.
*   **Speed-Quality Balance:** Choose from rapid or detailed 3D reconstruction methods.

## News
*   **Sep 02, 2025:** 🎉 Released a 5B model with low-VRAM mode, requiring only 12GB of VRAM!
*   **Aug 29, 2025:** 🎉 Gradio demo available! [See Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo)
*   **Aug 25, 2025:** 🎉 Low VRAM script released for 19G VRAM usage.
*   **Aug 12, 2025:** 🎉 Code, technical report, and project page released!

## Image-to-Scene Generation

| Image                                                                | Panoramic Video                                                  | 3D Scene                                                           |
| -------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_2.gif" height="150" width="300">      | <img src="asset/i2p/i2p_2_3D.gif" height="150">                    |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">    | <img src="asset/i2p/i2p_1_3D.gif" height="150">                   |

## Text-to-Scene Generation

| Text                                                             | Panoramic Video                                                  | 3D Scene                                                           |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall | <img src="asset/t2p/t2p_1.gif"  height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" > | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

**Related Project:** Explore Real-Time Interactive World Models with [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux with NVIDIA GPU.

```bash
# Clone the repository
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D

# Create a conda environment
conda create -n matrix3d python=3.10
conda activate matrix3d

# Install torch and torchvision (with GPU support, CUDA 12.4)
pip install torch==2.7.0 torchvision==0.22.0

# Run installation script
chmod +x install.sh
./install.sh
```

## Pretrained Models

| Model Name            | Description                                             | Download                                         |
| --------------------- | ------------------------------------------------------- | ------------------------------------------------ |
| Text2PanoImage      | Converts text prompts to panoramic images                   | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-480p      | Generates panoramic videos with 480p resolution.                   | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p      | Generates panoramic videos with 720p resolution.                   | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p-5B   | Generates panoramic videos with 720p resolution, with a small model.                   | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoLRM-480p        |  Low-Rank Model for quick 3D generation.                         | [Link](https://huggingface.co/Skywork/Matrix-3D) |

<!-- ## 📊 GPU VRAM Requirements -->
The minimum GPU VRAM requirement to run our pipeline is **16GB**.

| Model Name         | VRAM       | VRAM with low-vram mode |
| ------------------ | ---------- | ----------------------- |
| Text2PanoImage     | ~16GB      | -                       |
| PanoVideoGen-480p  | ~40GB      | ~15GB                   |
| PanoVideoGen-720p  | ~60GB      | ~19GB                   |
| PanoVideoGen-720p-5B| ~19GB      | ~12GB                  |
| PanoLRM-480p       | ~80GB      | -                       |

**Note:** The inference of PanoLRM requires a lot of VRAM, but is optional.  You can use the optimization-based reconstruction (see below) instead, which requires about 10GB VRAM.

## Usage

*   **🔧 Checkpoint Download**

```bash
python code/download_checkpoints.py
```

*   **🔥 One-command 3D World Generation**

```bash
./generate.sh
```

*   **🖼️ Step 1: Text/Image to Panorama Image**

From text prompt:

```bash
python code/panoramic_image_generation.py \
    --mode=t2p \
    --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
    --output_path="./output/example1"
```

From image input:

```bash
python code/panoramic_image_generation.py \
    --mode=i2p \
    --input_image_path="./data/image1.jpg" \
    --output_path="./output/example1"
```

If using your own panorama image, organize with prompt as:

```
./output/example1
└─ pano_img.jpg
└─ prompt.txt
```

*   **📹 Step 2: Generate Panoramic Video**

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
```
*You can choose a video resolution of 480 or 720.*

**Low VRAM Mode:**

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720 \
  --enable_vram_management
```

**5B Model:**

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720 \
  --use_5b_model
```

*   **🏡 Step 3: Extract 3D Scene**

Optimization-based reconstruction:

```bash
python code/panoramic_video_to_3DScene.py \
    --inout_dir="./output/example1" \
    --resolution=720
```

Feed-forward reconstruction:

```bash
python code/panoramic_video_480p_to_3DScene_lrm.py \
--video_path="./data/case1/sample_video.mp4" \
--pose_path='./data/case1/sample_cam.json' \
--out_path='./output/example2'
```

## 🎬 Create Your Own

| Movement Mode         | Trajectory              | Panoramic Video                   | 3D Scene                            |
| --------------------- | ----------------------- | --------------------------------- | ----------------------------------- |
| S-curve Travel        | <img src="asset/movement/s.PNG"  height="120"  width="120"  >      | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" >     |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

Configure movement modes in `code/panoramic_image_to_video.py` using `--movement_mode`. Custom camera trajectories can also be used:

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

Refer to `./data/test_cameras/test_cam_front.json` and `code/generate_example_camera.py` for trajectory creation.

## 🖱️ Gradio Demo

Run for the Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

## 📚 Citation

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}
```

---

## 🤝 Acknowledgements

*   FLUX.1
*   Wan2.1
*   WorldGen
*   MoGe
*   nvdiffrast
*   gaussian-splatting
*   StableSR
*   VEnhancer

## 📧 Contact

Open an issue for questions or feature requests.