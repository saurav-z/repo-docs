# Matrix-3D: Generate Explorable 3D Worlds from Text and Images

**Create immersive, 360-degree explorable 3D worlds with Matrix-3D, transforming your ideas into interactive experiences!**

[![](https://img.shields.io/badge/GitHub-SkyworkAI/Matrix--3D-blue?style=flat&logo=github)](https://github.com/SkyworkAI/Matrix-3D)

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="600" style="margin-bottom: 5px;"/>  
</div>

<div align="center">
  <a href="https://matrix-3d.github.io/"><img src="https://img.shields.io/badge/📄-Project_Page-orange" alt="Project Page"></a>
  <a href="https://huggingface.co/Skywork/Matrix-3D"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model"></a>
  <a href="https://arxiv.org/pdf/2508.08086"><img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Technical Report"></a>
  <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version">
</div>

## Key Features:

*   **360° Scene Generation:** Create expansive, omnidirectional scenes for complete exploration.
*   **Text and Image Input:** Generate 3D worlds from text prompts or input images.
*   **Customizable Trajectories:** Define your own camera paths for unique video experiences.
*   **High-Quality Results:** Leverage advanced models for diverse and detailed 3D scene generation.
*   **Speed-Quality Balance:** Choose between rapid or detailed 3D reconstruction methods.
*   **Low VRAM Option:** Run the video generation process with as little as 12GB of VRAM, a more accessible option!
*   **5B Model**: Utilize our new, fast, and efficient 5B model to generate videos!

## What's New:

*   **September 02, 2025:** Released a 5B model with low VRAM usage (12GB).
*   **August 29, 2025:** Launched a Gradio demo for easy visualization.
*   **August 25, 2025:** Provided a script to run the generation process with 19GB VRAM.
*   **August 12, 2025:** Publicly released the code, technical report, and project page.

## Visual Examples:

### Image-to-Scene Generation

| Image                                                                                                                                                                     | Panoramic Video                                                                   | 3D Scene                                                                          |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br>                                                                                             | <img src="asset/i2p/i2p_2.gif" height="150" width="300">                            | <img src="asset/i2p/i2p_2_3D.gif" height="150">                                   |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br>                                                                                             | <img src="asset/i2p/i2p_1.gif" height="150" width="300">                            | <img src="asset/i2p/i2p_1_3D.gif" height="150">                                   |

### Text-to-Scene Generation

| Text                                                                                                                                                                      | Panoramic Video                                                                   | 3D Scene                                                                          |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">  <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif" height="150" width="300">                            | <img src="asset/t2p/t2p_1_3D.gif" height="150">                                   |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">  <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</div> | <img src="asset/t2p/t2p_2.gif" height="150" width="300">                            | <img src="asset/t2p/t2p_2_3D.gif" height="150">                                   |

## Installation:

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

3.  **Install PyTorch and torchvision (with CUDA support):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```

4.  **Run the installation script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models:

| Model Name           | Description                       | Download                                         |
| :------------------- | :-------------------------------- | :--------------------------------------------- |
| Text2PanoImage       | Text-to-Panoramic Image           | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-480p    | Panoramic Video Generation (480p) | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p    | Panoramic Video Generation (720p) | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p-5B | Panoramic Video Generation (720p, 5B)  | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoLRM-480p         | Panoramic LRM (480p)             | [Link](https://huggingface.co/Skywork/Matrix-3D) |

## GPU VRAM Requirements:

| Model Name          | VRAM            | VRAM with Low-VRAM Mode |
| :------------------ | :-------------- | :----------------------- |
| Text2PanoImage      | \~16GB          | -                        |
| PanoVideoGen-480p   | \~40GB          | \~15GB                   |
| PanoVideoGen-720p   | \~60GB          | \~19GB                   |
| PanoVideoGen-720p-5B| \~19GB          | \~12GB                   |
| PanoLRM-480p        | \~80GB          | -                        |

**Note:**  PanoLRM inference is optional and uses significant VRAM. Use the optimization-based reconstruction method for a lighter footprint (approx. 10GB VRAM).

## Usage:

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

            Ensure your input image and prompt are saved in the following structure:
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

        *   **Low VRAM Mode:** (For devices with limited VRAM):
            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --enable_vram_management # enable this to allow model to run on devices with 19G vram.
            ```
        *   **5B Model:** (For faster generation and lower VRAM usage):
            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --use_5b_model # enable this to generate video with light-weight 5B model.
            ```

    *   **Step 3: Extract 3D Scene:**

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

## Create Your Own:

### Movement Modes & Trajectories:

| Movement Mode       | Trajectory                                 | Panoramic Video                                      | 3D Scene                                               |
| :------------------ | :----------------------------------------- | :--------------------------------------------------- | :----------------------------------------------------- |
| S-curve Travel      | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300">    | <img src="asset/movement/s_3D.gif" height="150" >    |
| Forward on the Right| <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

*   **Movement Modes:**  `Straight Travel`, `S-curve Travel`, and `Forward on the Right`.  Configure with `--movement_mode` in `code/panoramic_image_to_video.py`.
*   **Custom Camera Trajectory:**  Provide your own camera path in `.json` format.

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
      --json_path YOUR_TRAJECTORY_FILE.json
    ```
    Refer to `./data/test_cameras/test_cam_front.json` for the correct camera matrix format (world-to-camera in OpenCV format).  Use `code/generate_example_camera.py` to generate your own camera trajectory files.

## Gradio Demo:

Run the Gradio demo for easy visualization:

```bash
python code/matrix.py --max_gpus=1
```

*   **GPU Configuration:**
    *   `--max_gpus=1`: Supports text-video-3D generation (requires a GPU with at least 62GB of memory).
    *   `--max_gpus=N, N≥2`: Supports both text-video-3D and image-video-3D generation workflows.  Allocate GPUs based on your hardware.

## Citation:

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}
```

## Acknowledgements:

This project builds upon the following resources:

*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact:

For questions or feature requests, please open an issue on GitHub.