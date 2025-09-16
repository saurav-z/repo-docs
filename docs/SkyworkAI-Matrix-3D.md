# Matrix-3D: Generate and Explore Omnidirectional 3D Worlds

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="800" style="margin-bottom: 5px;"/>
  <br>
  *Unleash your imagination and explore limitless 3D worlds with Matrix-3D!*
</div>

<div align="center">
  [![Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
  [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
  ![Badge](https://img.shields.io/badge/version-v0.1.0-green)
  [![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)
</div>

[Link to original repository:](https://github.com/SkyworkAI/Matrix-3D)

## Key Features

*   **Expansive Scene Generation:** Create immersive 360-degree explorable environments beyond typical scene generation approaches.
*   **High Controllability:** Generate scenes from both text and image inputs with customizable trajectories for versatile creative exploration.
*   **Strong Generalization:** Leverage self-developed 3D data and video model priors for diverse, high-quality 3D scene generation.
*   **Speed-Quality Balance:** Choose between rapid and detailed 3D reconstruction methods based on your needs.

## What's New

*   **September 2, 2025:** 🎉 A 5B model is now available, with a low-VRAM mode requiring only 12GB of VRAM!
*   **August 29, 2025:** 🎉 Explore Matrix-3D interactively with our new [Gradio demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo).
*   **August 25, 2025:** 🎉 Run the generation process with a reduced VRAM footprint, thanks to our new [script](#lowvram) (19GB VRAM).
*   **August 12, 2025:** 🎉 Release of code, technical report, and project page!

## Image-to-Scene Generation

| Image                                                                                    | Panoramic Video                                                                               | 3D Scene                                                                      |
| :--------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;">                 | <img src="asset/i2p/i2p_2.gif"  height="150" width="300">                                    | <img src="asset/i2p/i2p_2_3D.gif" height="150">                               |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;">                 | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">                                    | <img src="asset/i2p/i2p_1_3D.gif" height="150">                               |

## Text-to-Scene Generation

| Text                                                                                                                                                                    | Panoramic Video                                                                            | 3D Scene                                                                     |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| <span style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">A floating island with a waterfall</span> | <img src="asset/t2p/t2p_1.gif"  height="150" width="300">                                   | <img src="asset/t2p/t2p_1_3D.gif" height="150">                              |
| <span style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">an impressionistic winter landscape</span>   | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" >                                  | <img src="asset/t2p/t2p_2_3D.gif" height="150">                              |

**Related Project:** Explore Real-Time Interactive Long-Sequence World Models with [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux systems with NVIDIA GPUs.

1.  **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
    cd Matrix-3D
    ```

2.  **Create and Activate a Conda Environment:**

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

| Model Name             | Description                                  | Download                                                          |
| :--------------------- | :------------------------------------------- | :---------------------------------------------------------------- |
| Text2PanoImage         | -                                            | [Link](https://huggingface.co/Skywork/Matrix-3D)                  |
| PanoVideoGen-480p      | -                                            | [Link](https://huggingface.co/Skywork/Matrix-3D)                  |
| PanoVideoGen-720p      | -                                            | [Link](https://huggingface.co/Skywork/Matrix-3D)                  |
| PanoVideoGen-720p-5B   | -                                            | [Link](https://huggingface.co/Skywork/Matrix-3D)                  |
| PanoLRM-480p           | -                                            | [Link](https://huggingface.co/Skywork/Matrix-3D)                  |

## GPU VRAM Requirements

The minimum GPU VRAM required is 16GB.  For reduced VRAM usage, see the [Low VRAM Mode](#lowvram) section.

| Model Name           | VRAM (Approx.) | VRAM with Low-VRAM Mode |
| :------------------- | :------------- | :--------------------- |
| Text2PanoImage       | \~16GB         | -                      |
| PanoVideoGen-480p    | \~40GB         | \~15GB                 |
| PanoVideoGen-720p    | \~60GB         | \~19GB                 |
| PanoVideoGen-720p-5B | \~19GB         | \~12GB                 |
| PanoLRM-480p         | \~80GB         | -                      |

**Note:** PanoLRM reconstruction requires substantial VRAM, consider the optimization-based method (see Usage) which uses around 10GB VRAM.

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

        *   If using a custom panorama image, structure your files like this in the output folder:
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

        *   Adjust `--resolution` to `480` or `720` for different video sizes. The process may take about an hour with an A800 GPU, and you can accelerate with multi-GPU (set `VISIBLE_GPU_NUM`).

        <span id="lowvram">**Low VRAM Mode:**</span>  For devices with limited VRAM:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --enable_vram_management
        ```

        <span id="5B">**5B Model:**</span>  For faster generation with lower VRAM usage:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --use_5b_model
        ```

    3.  **Extract 3D Scene:**
        *   **Optimization-based Reconstruction (High Quality):**

            ```bash
            python code/panoramic_video_to_3DScene.py \
               --inout_dir="./output/example1" \
               --resolution=720
            ```
        *   **Feed-Forward Reconstruction (Efficient):**

            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
            ```

## Create Your Own Scenes

| Movement Mode                               | Trajectory                                                         | Panoramic Video                                                            | 3D Scene                                                                     |
| :------------------------------------------ | :----------------------------------------------------------------- | :------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| <span style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">S-curve Travel</span> | <img src="asset/movement/s.PNG"  height="120"  width="120"  >                 | <img src="asset/movement/s.gif" height="150"  width="300">                   | <img src="asset/movement/s_3D.gif" height="150" >                                |
| <span style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">Forward on the Right</span> | <img src="asset/movement/forward.PNG"  height="120"  width="120" >            | <img src="asset/movement/forward.gif" height="150" width="300">               | <img src="asset/movement/forward_3D.gif" height="150">                               |

*   **Movement Modes:** `Straight Travel`, `S-curve Travel`, and `Forward on the Right`. Configure using `--movement_mode` in `code/panoramic_image_to_video.py`.
*   **Custom Camera Trajectories:** Use a .json file.

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720 \
      --json_path YOUR_TRAJECTORY_FILE.json
    ```

    *   Use  `code/generate_example_camera.py` to generate your own trajectories (in OpenCV world-to-camera format).

## Gradio Demo

```bash
python code/matrix.py --max_gpus=1
```

*   **GPU Configuration:**
    *   `--max_gpus=1`: Text-video-3D generation (requires ~62GB of memory).
    *   `--max_gpus=N, N≥2`: Text-video-3D and image-video-3D workflows (allocate GPUs based on your hardware).

## Citation

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

This project builds on the following:

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

Please open an issue if you have any questions or feature requests.