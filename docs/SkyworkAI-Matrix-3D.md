# Matrix-3D: Generate Immersive, Explorable 3D Worlds from Text and Images

[View the original repository](https://github.com/SkyworkAI/Matrix-3D)

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">
  <a href="https://matrix-3d.github.io/"><img src="https://img.shields.io/badge/📄-Project_Page-orange" alt="Project Page"></a>
  <a href="https://huggingface.co/Skywork/Matrix-3D"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model"></a>
  <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version">
  <a href="https://arxiv.org/pdf/2508.08086"><img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Technical Report"></a>
</div>

Matrix-3D revolutionizes 3D world generation by transforming text prompts and images into explorable, omnidirectional 3D environments.

## Key Features

*   **Omnidirectional Exploration:** Generate complete 360-degree explorable scenes.
*   **Versatile Input:** Supports both text-to-scene and image-to-scene generation.
*   **Customizable:** Control trajectories and extend scenes with infinite possibilities.
*   **High-Quality & Diverse:** Generate diverse, high-quality 3D scenes based on advanced models.
*   **Speed-Quality Balance:** Offers both rapid and detailed 3D reconstruction methods.

## What's New

*   **September 2, 2025:** Released a 5B model with low-VRAM mode, requiring only 12GB of VRAM!
*   **August 29, 2025:** Launched a [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) for easy exploration.
*   **August 25, 2025:**  Provided a [script](#lowvram) for running the generation process with 19G VRAM.
*   **August 12, 2025:**  Released code, a technical report, and a project page.

## Showcase: Image-to-Scene Generation

| Image                                     | Panoramic Video                                    | 3D Scene                                     |
| :---------------------------------------- | :------------------------------------------------- | :------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_2.gif" height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150"> |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_1.gif" height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150"> |

## Showcase: Text-to-Scene Generation

| Text                                  | Panoramic Video                                    | 3D Scene                                     |
| :------------------------------------ | :------------------------------------------------- | :------------------------------------------- |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative;">A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif" height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative;">an impressionistic winter landscape</div> | <img src="asset/t2p/t2p_2.gif" height="150" width="300"> | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

**Related Project:** Explore Real-Time Interactive Long-Sequence World Models at [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux systems with NVIDIA GPUs.

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
3.  **Install Dependencies:**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0  # Ensure CUDA 12.4 is supported
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name              | Description                               | Download                                             |
| :--------------------- | :---------------------------------------- | :--------------------------------------------------- |
| Text2PanoImage        |  Generates panorama images from text.  | [Link](https://huggingface.co/Skywork/Matrix-3D)   |
| PanoVideoGen-480p     |  Generates 480p panoramic videos.     | [Link](https://huggingface.co/Skywork/Matrix-3D)   |
| PanoVideoGen-720p     |  Generates 720p panoramic videos.     | [Link](https://huggingface.co/Skywork/Matrix-3D)   |
| PanoVideoGen-720p-5B  |  Fast video generation with 5B model.  | [Link](https://huggingface.co/Skywork/Matrix-3D)   |
| PanoLRM-480p          |  Low-resource model for panorama.     | [Link](https://huggingface.co/Skywork/Matrix-3D)   |

## GPU VRAM Requirements

The following table details the approximate VRAM usage.

| Model Name              | VRAM        | VRAM with low-vram mode |
| :--------------------- | :---------- | :--------------------- |
| Text2PanoImage        | ~16 GB      | -                      |
| PanoVideoGen-480p     | ~40 GB      | ~15 GB                 |
| PanoVideoGen-720p     | ~60 GB      | ~19 GB                 |
| PanoVideoGen-720p-5B  | ~19 GB      | ~12 GB                 |
| PanoLRM-480p          | ~80 GB      | -                      |

**Note:** PanoLRM requires significant VRAM but is optional.  The optimization-based reconstruction method requires approximately 10GB of VRAM.

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

    *   **Step 1: Text/Image to Panorama Image:**
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
        *   Prepare your own panorama image and prompt in the following structure before proceeding to Step 2:

            ```
            ./output/example1
            └─ pano_img.jpg
            └─ prompt.txt
            ```

    *   **Step 2: Generate Panoramic Video:**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```
        You can set resolution as 480 or 720. 720p video generation takes approximately an hour on an A800 GPU. Use `VISIBLE_GPU_NUM` to enable multi-GPU inference.

        *   **Low VRAM Mode:**

            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --enable_vram_management  # enable this to allow model to run on devices with 19G vram.
            ```

        *   **5B Model:**

            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --use_5b_model  # enable this to generate video with light-weight 5B model.
            ```

    *   **Step 3: Extract 3D Scene:**

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

## Create Your Own

| Movement Mode         | Trajectory                     | Panoramic Video                  | 3D Scene                      |
| :-------------------- | :----------------------------- | :------------------------------- | :---------------------------- |
| S-curve Travel       | <img src="asset/movement/s.PNG"  height="120"  width="120"  >      | <img src="asset/movement/s.gif" height="150"  width="300">      | <img src="asset/movement/s_3D.gif" height="150" >             |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

*   Configure movement modes (`Straight Travel`, `S-curve Travel`, `Forward on the Right`) using `--movement_mode` in `code/panoramic_image_to_video.py`.
*   Use custom camera trajectories in .json format.

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
      --json_path YOUR_TRAJECTORY_FILE.json
    ```

    *   Refer to `./data/test_cameras/test_cam_front.json` for camera matrix format (world to camera, OpenCV format). Use `code/generate_example_camera.py` to generate trajectories.

## Gradio Demo

Launch the Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

*   **GPU Configuration:**
    *   `--max_gpus=1`: Text-video-3D generation (62+ GB VRAM recommended).
    *   `--max_gpus=N` (N ≥ 2):  Text-video-3D and image-video-3D, adjust GPU allocation accordingly.

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