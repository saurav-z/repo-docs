# Matrix-3D: Generate and Explore Omnidirectional 3D Worlds

Create breathtaking, explorable 3D worlds from text and images with **Matrix-3D**, a cutting-edge platform. Discover a new dimension of immersive content creation!  Explore the original repo [here](https://github.com/SkyworkAI/Matrix-3D).

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">

[![📄 Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
![Badge](https://img.shields.io/badge/version-v0.1.0-green)
[![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)

</div>

## Key Features

*   **Omnidirectional Exploration:** Generate 360-degree explorable scenes.
*   **Text and Image Input:** Easily create scenes from text prompts or images.
*   **High Controllability:** Customize trajectories and enjoy infinite extensibility.
*   **Strong Generalization:** Built on self-developed 3D data and video model priors.
*   **Speed-Quality Balance:** Choose from two panoramic 3D reconstruction methods.

## What's New?

*   **September 2, 2025:** 🎉  5B Model with Low-VRAM Mode (12GB VRAM required!)
*   **August 29, 2025:** 🎉  [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) available!
*   **August 25, 2025:** 🎉  [Script](#lowvram) for 19GB VRAM video generation released!
*   **August 12, 2025:** 🎉  Code, technical report, and project page released!

## Image-to-Scene Generation

| Image                                                                                                                                   | Panoramic Video                                         | 3D Scene                                                  |
| :-------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------ | :-------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br> | <img src="asset/i2p/i2p_2.gif" height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150">           |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br> | <img src="asset/i2p/i2p_1.gif" height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150">           |

## Text-to-Scene Generation

| Text                                                                                                                                                                                                                   | Panoramic Video                                         | 3D Scene                                                  |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------ | :-------------------------------------------------------- |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden; font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3);">A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif" height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150">           |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden; font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3);">an impressionistic winter landscape</div>     | <img src="asset/t2p/t2p_2.gif" height="150" width="300"> | <img src="asset/t2p/t2p_2_3D.gif" height="150">           |

## Related Project

Explore Real-Time Interactive World Models: [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2)

## Installation

Tested on Linux systems with NVIDIA GPUs.

```bash
# Clone the repository
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D

# Create a new conda environment
conda create -n matrix3d python=3.10
conda activate matrix3d

# Install PyTorch and TorchVision (with CUDA 12.4 support)
pip install torch==2.7.0 torchvision==0.22.0

# Run the installation script
chmod +x install.sh
./install.sh
```

## Pretrained Models

| Model Name           | Description            | Download                                            |
| :------------------- | :--------------------- | :-------------------------------------------------- |
| Text2PanoImage       | Text to Panoramic Image | [Link](https://huggingface.co/Skywork/Matrix-3D)     |
| PanoVideoGen-480p    | 480p Video Generation  | [Link](https://huggingface.co/Skywork/Matrix-3D)     |
| PanoVideoGen-720p    | 720p Video Generation  | [Link](https://huggingface.co/Skywork/Matrix-3D)     |
| PanoVideoGen-720p-5B | 720p 5B Model          | [Link](https://huggingface.co/Skywork/Matrix-3D)     |
| PanoLRM-480p         | Panoramic LRM          | [Link](https://huggingface.co/Skywork/Matrix-3D)     |

<!-- ## 📊 GPU VRAM Requirements -->

The minimum GPU VRAM required is 16GB. Utilize the [script](#lowvram) for low-VRAM video generation.

| Model Name          | VRAM     | VRAM with low-vram mode |
| :------------------ | :------- | :---------------------- |
| Text2PanoImage      | ~16 GB   | -                       |
| PanoVideoGen-480p   | ~40 GB   | ~15 GB                  |
| PanoVideoGen-720p   | ~60 GB   | ~19 GB                  |
| PanoVideoGen-720p-5B| ~19 GB   | ~12 GB                  |
| PanoLRM-480p        | ~80 GB   | -                       |

**Note:** PanoLRM inference requires significant VRAM; optimization-based reconstruction (see below) requires ~10GB.

## Usage

-   🔧 **Checkpoint Download:**

    ```bash
    python code/download_checkpoints.py
    ```

-   🔥 **One-command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

-   🖼️ **Step 1: Text/Image to Panorama Image:**

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

        Output panorama images in `./output/example1`.  If using your own image, structure it with a `pano_img.jpg` and `prompt.txt` file.

-   📹 **Step 2: Generate Panoramic Video:**

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
    ```

    Adjust `--resolution` to 480 or 720. A 720p video generation takes approximately one hour on an A800 GPU.  Use `VISIBLE_GPU_NUM` for multi-GPU acceleration.

    <span id="lowvram">**Low VRAM Mode:**</span>

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720 \
      --enable_vram_management
    ```

    <span id="5B">**5B Model:**</span>

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720 \
      --use_5b_model
    ```

-   🏡 **Step 3: Extract 3D Scene:**

    *   **Optimization-based Reconstruction:**

        ```bash
         python code/panoramic_video_to_3DScene.py \
            --inout_dir="./output/example1" \
            --resolution=720
        ```
        Output: `output/example1/generated_3dgs_opt.ply`.
    *   **Feed-forward Reconstruction:**

        ```bash
        python code/panoramic_video_480p_to_3DScene_lrm.py \
        --video_path="./data/case1/sample_video.mp4" \
        --pose_path='./data/case1/sample_cam.json' \
        --out_path='./output/example2'
        ```
        Output: 3D scene (.ply) and rendered perspective videos in `output/example2`.

## 🎬 Create Your Own

| Movement Mode           | Trajectory                                  | Panoramic Video                         | 3D Scene                                  |
| :---------------------- | :------------------------------------------ | :-------------------------------------- | :------------------------------------------ |
| S-curve Travel          | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| Forward on the Right    | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

Movement Modes: `Straight Travel`, `S-curve Travel`, and `Forward on the Right` (configured in `--movement_mode` in `code/panoramic_image_to_video.py`).

Use custom camera trajectories (in .json format)

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

Camera matrices are world-to-camera, OpenCV format. Refer to `./data/test_cameras/test_cam_front.json` and generate trajectories with `code/generate_example_camera.py`.

## 🖱️ Gradio Demo

```bash
python code/matrix.py --max_gpus=1
```

GPU Configuration:
*   Single GPU (--max_gpus=1): Supports text-video-3D generation (62GB+ memory recommended).
*   Multiple GPUs (--max_gpus=N, N≥2): Supports both text-video-3D and image-video-3D generation.

## 📚 Citation

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}

@article{dong2025panolora,
  title     = {PanoLora: Bridging Perspective and Panoramic Video Generation with LoRA Adaptation},
  author    = {Zeyu Dong and Yuyang Yin and Yuqi Li and Eric Li and Hao-Xiang Guo and Yikai Wang},
  journal   = {arXiv preprint arXiv:2509.11092},
  year      = {2025}
}
```

---

## 🤝 Acknowledgements

This project is built upon the following resources; please cite them if you find them helpful:
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