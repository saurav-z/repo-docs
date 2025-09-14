<h1 align="center">
    Matrix-3D: Generate Interactive 3D Worlds from Text or Images
</h1>

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">

[![📄 Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
![Badge](https://img.shields.io/badge/version-v0.1.0-green)
[![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)
</div>

**Matrix-3D empowers you to create expansive, explorable 3D worlds from text prompts or images, offering unparalleled freedom in scene generation.** ([See the original repository](https://github.com/SkyworkAI/Matrix-3D))

## Key Features

*   **Omnidirectional Exploration:** Generate and explore 360-degree scenes.
*   **Text-to-Scene & Image-to-Scene:**  Convert text descriptions or images into interactive 3D environments.
*   **High Controllability:** Customize scenes with text, images, and flexible camera trajectories.
*   **Strong Generalization:** Leverage self-developed 3D and video model priors for diverse, high-quality scenes.
*   **Speed-Quality Balance:** Choose from rapid or detailed 3D reconstruction methods.

## What's New

*   **September 2, 2025:**  🎉  5B model with low-VRAM mode released (12G VRAM requirement).
*   **August 29, 2025:**  🎉  Gradio demo available.
*   **August 25, 2025:**  🎉  Script for 19G VRAM generation.
*   **August 12, 2025:**  🎉  Code, technical report, and project page launched.

## Image-to-Scene Generation

<table border="1">
<tr>
  <th>Image</th>
  <th>Panoramic Video</th>
  <th>3D Scene</th>
</tr>
<tr>
  <td width="210" height="150" style="
  padding: 15px;
  border: 1px solid rgba(168,237,234,0.5);
  border-radius: 8px;
  background-color: rgba(10,20,30,0.7);
  position: relative;
  text-align: center;
  vertical-align: top;
  font-family: 'Palatino', 'Georgia', serif;">
  
  <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br>
</td>
  <td><img src="asset/i2p/i2p_2.gif"  height="150" width="300"></td>
  <td><img src="asset/i2p/i2p_2_3D.gif" height="150"></td>
</tr>
<tr>
  <th>Image</th>
  <th>Panoramic Video</th>
  <th>3D Scene</th>
</tr>
<tr>
  <td width="210" height="150" style="
  padding: 15px;
  border: 1px solid rgba(168,237,234,0.5);
  border-radius: 8px;
  background-color: rgba(10,20,30,0.7);
  position: relative;
  text-align: center;
  vertical-align: top;
  font-family: 'Palatino', 'Georgia', serif;">
  
  <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br>
</td>
  <td><img src="asset/i2p/i2p_1.gif"  height="150" width="300"></td>
  <td><img src="asset/i2p/i2p_1_3D.gif" height="150"></td>
</tr>
</table>

## Text-to-Scene Generation

<table border="1">
<tr>
  <th>Text</th>
  <th>Panoramic Video</th>
  <th>3D Scene</th>
</tr>
<tr>
  <th width="200" style="
  font-family: 'Palatino', 'Georgia', serif;
  font-size: 1.3em;
  color: transparent;
  background: 
    linear-gradient(45deg, 
      #a8edea 0%, 
      #fed6e3 50%, 
      #a8edea 100%);
  -webkit-background-clip: text;
  background-clip: text;
  text-shadow: 
    0 0 5px rgba(168,237,234,0.3),
    0 0 10px rgba(254,214,227,0.3);
  padding: 15px;
  border: 1px solid rgba(168,237,234,0.5);
  border-radius: 8px;
  background-color: rgba(10,20,30,0.7);
  position: relative;
  overflow: hidden;
">
  <div style="
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: 
      radial-gradient(circle at 20% 30%, 
        rgba(254,214,227,0.1) 0%, 
        transparent 40%),
      radial-gradient(circle at 80% 70%, 
        rgba(168,237,234,0.1) 0%, 
        transparent 40%);
    z-index: -1;
  "></div>A floating island with a waterfall</th>
  
  <td><img src="asset/t2p/t2p_1.gif"  height="150" width="300"></td>
  <td><img src="asset/t2p/t2p_1_3D.gif" height="150"></td>
</tr>
<tr>
  <th>Text</th>
  <th>Panoramic Video</th>
  <th>3D Scene</th>
</tr>
<tr>
  <th width="200" style="
  font-family: 'Palatino', 'Georgia', serif;
  font-size: 1.3em;
  color: transparent;
  background: 
    linear-gradient(45deg, 
      #a8edea 0%, 
      #fed6e3 50%, 
      #a8edea 100%);
  -webkit-background-clip: text;
  background-clip: text;
  text-shadow: 
    0 0 5px rgba(168,237,234,0.3),
    0 0 10px rgba(254,214,227,0.3);
  padding: 15px;
  border: 1px solid rgba(168,237,234,0.5);
  border-radius: 8px;
  background-color: rgba(10,20,30,0.7);
  position: relative;
  overflow: hidden;
">
  <div style="
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: 
      radial-gradient(circle at 20% 30%, 
        rgba(254,214,227,0.1) 0%, 
        transparent 40%),
      radial-gradient(circle at 80% 70%, 
        rgba(168,237,234,0.1) 0%, 
        transparent 40%);
    z-index: -1;
  "></div>an impressionistic winter landscape</th>
  <td><img src="asset/t2p/t2p_2.gif"  height="150"  width="300" ></td>
  <td><img src="asset/t2p/t2p_2_3D.gif" height="150"></td>
</tr>
</table>

## Related Projects

Explore real-time interactive world models with [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux systems with NVIDIA GPUs.

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

3.  **Install PyTorch and torchvision (with CUDA 12.4 support):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```

4.  **Run the installation script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name | Description | Download |
| :---------: | :----------: |  :-: |
| Text2PanoImage | Text-to-Panoramic Image | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-480p | Panoramic Video Generation (480p) | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p | Panoramic Video Generation (720p) | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p-5B | Panoramic Video Generation (720p, 5B Model) | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoLRM-480p | Panoramic Light Reconstruction Model (480p) | [Link](https://huggingface.co/Skywork/Matrix-3D) |

## GPU VRAM Requirements

| Model Name            | VRAM Usage | VRAM with Low-VRAM Mode |
| --------------------- | ---------- | ---------------------- |
| Text2PanoImage        | ~16GB      | -                      |
| PanoVideoGen-480p     | ~40GB      | ~15GB                  |
| PanoVideoGen-720p     | ~60GB      | ~19GB                  |
| PanoVideoGen-720p-5B  | ~19GB      | ~12GB                  |
| PanoLRM-480p          | ~80GB      | -                      |

**Note:** PanoLRM reconstruction is optional.  Optimization-based reconstruction uses ~10GB VRAM.

## Usage

1.  **Download Checkpoints:**

    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

    Or generate step-by-step:

3.  **Step 1: Text/Image to Panoramic Image:**

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

        Place custom images with prompt files in the following structure in the `output` directory:

        ```
        ./output/example1
        └─ pano_img.jpg
        └─ prompt.txt
        ```

4.  **Step 2: Generate Panoramic Video:**

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
    ```

    *   Adjust `--resolution` to `480` or `720`.
    *   Generate with low-VRAM mode:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --enable_vram_management
        ```

    *   Generate with 5B model:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --use_5b_model
        ```

5.  **Step 3: Extract 3D Scene:**

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

## Create Your Own Trajectories & Scenes

<table border="1">
<tr>
  <th>Movement Mode</th>
  <th>Trajectory</th>
  <th>Panoramic Video</th>
  <th>3D Scene</th>
</tr>
<tr>
  <th width="30" style="
  font-family: 'Palatino', 'Georgia', serif;
  font-size: 1.3em;
  color: transparent;
  background: 
    linear-gradient(45deg, 
      #a8edea 0%, 
      #fed6e3 50%, 
      #a8edea 100%);
  -webkit-background-clip: text;
  background-clip: text;
  text-shadow: 
    0 0 5px rgba(168,237,234,0.3),
    0 0 10px rgba(254,214,227,0.3);
  padding: 15px;
  border: 1px solid rgba(168,237,234,0.5);
  border-radius: 8px;
  background-color: rgba(10,20,30,0.7);
  position: relative;
  overflow: hidden;
">
  <div style="
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: 
      radial-gradient(circle at 20% 30%, 
        rgba(254,214,227,0.1) 0%, 
        transparent 40%),
      radial-gradient(circle at 80% 70%, 
        rgba(168,237,234,0.1) 0%, 
        transparent 40%);
    z-index: -1;
  "></div>S-curve Travel</th>
  <td><img src="asset/movement/s.PNG"  height="120"  width="120"  ></td>
  <td><img src="asset/movement/s.gif" height="150"  width="300"></td>
  <td><img src="asset/movement/s_3D.gif" height="150" ></td>
</tr>
<tr>
  <th width="30" style="
  font-family: 'Palatino', 'Georgia', serif;
  font-size: 1.3em;
  color: transparent;
  background: 
    linear-gradient(45deg, 
      #a8edea 0%, 
      #fed6e3 50%, 
      #a8edea 100%);
  -webkit-background-clip: text;
  background-clip: text;
  text-shadow: 
    0 0 5px rgba(168,237,234,0.3),
    0 0 10px rgba(254,214,227,0.3);
  padding: 15px;
  border: 1px solid rgba(168,237,234,0.5);
  border-radius: 8px;
  background-color: rgba(10,20,30,0.7);
  position: relative;
  overflow: hidden;
">
  <div style="
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: 
      radial-gradient(circle at 20% 30%, 
        rgba(254,214,227,0.1) 0%, 
        transparent 40%),
      radial-gradient(circle at 80% 70%, 
        rgba(168,237,234,0.1) 0%, 
        transparent 40%);
    z-index: -1;
  "></div>Forward on the Right</th>
  <td><img src="asset/movement/forward.PNG"  height="120"  width="120" ></td>
  <td><img src="asset/movement/forward.gif" height="150" width="300"></td>
  <td><img src="asset/movement/forward_3D.gif" height="150"></td>
</tr>
</table>

*   **Movement Modes:**  Choose from `Straight Travel`, `S-curve Travel`, or `Forward on the Right` via the `--movement_mode` parameter in `code/panoramic_image_to_video.py`.
*   **Custom Trajectories:** Provide your camera trajectory in JSON format using the `--json_path YOUR_TRAJECTORY_FILE.json` argument.  Refer to `./data/test_cameras/test_cam_front.json` for example and `code/generate_example_camera.py` for creating your own.  Camera matrices should be world-to-camera in OpenCV format.

## Gradio Demo

Launch the Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

*   **GPU Configuration:**
    *   Single GPU (`--max_gpus=1`):  Text-video-3D workflow only. Requires at least 62GB of GPU memory.
    *   Multiple GPUs (`--max_gpus=N, N≥2`):  Supports text-video-3D and image-video-3D workflows.

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

This project utilizes code from:

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

For questions or feature requests, please [open an issue](https://github.com/SkyworkAI/Matrix-3D/issues).