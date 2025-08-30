# Matrix-3D: Generate Omnidirectional 3D Worlds from Text and Images

Unleash your imagination and explore boundless 3D worlds! Matrix-3D leverages cutting-edge AI to generate immersive, explorable 3D environments from simple text prompts or images. [Explore the original repository](https://github.com/SkyworkAI/Matrix-3D) to dive deeper!

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

*   **Text-to-3D & Image-to-3D:** Seamlessly generate 3D scenes from text descriptions or images.
*   **Omnidirectional Exploration:** Experience 360-degree, free exploration of generated worlds.
*   **High Controllability:** Customize scenes with text prompts, image inputs, and flexible camera trajectories.
*   **Large-Scale Scene Generation:** Generate broader and more expansive scenes compared to existing methods.
*   **Strong Generalization:** Built upon advanced 3D data and video model priors for diverse, high-quality scene generation.
*   **Speed-Quality Balance:** Utilize two 3D reconstruction methods (optimization-based and feed-forward) for rapid and detailed results.

## What's New

*   **August 29, 2025:** [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) released!
*   **August 25, 2025:** [Low VRAM script](#lowvram) for 19G VRAM support!
*   **August 12, 2025:** Code, technical report, and project page released!

## Image-to-Scene Generation Examples

| Image                                                                                                                              | Panoramic Video                                                                                                                                     | 3D Scene                                                                                                                         |
| :--------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;">                                                              | <img src="asset/i2p/i2p_2.gif"  height="150" width="300">                                                                                               | <img src="asset/i2p/i2p_2_3D.gif" height="150">                                                                              |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;">                                                              | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">                                                                                               | <img src="asset/i2p/i2p_1_3D.gif" height="150">                                                                              |

## Text-to-Scene Generation Examples

| Text                                                                                                                                                                                                                                                                  | Panoramic Video                                                                                                                                     | 3D Scene                                                                                                                         |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif"  height="150" width="300">                                                                                               | <img src="asset/t2p/t2p_1_3D.gif" height="150">                                                                              |
| <div style="padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</div> | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" >                                                                                                | <img src="asset/t2p/t2p_2_3D.gif" height="150">                                                                              |

**Related Project**: Explore Real-Time Interactive Long-Sequence World Models at [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

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

3.  **Install Dependencies (with GPU Support):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0  #CUDA 12.4 is recommended
    ```

4.  **Run the Installation Script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name         | Description             | Download                                      |
| :------------------ | :---------------------- | :-------------------------------------------- |
| Text2PanoImage     | Text-to-Panorama Image  | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-480p  | Panorama Video (480p)   | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p  | Panorama Video (720p)   | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoLRM-480p       | Panorama LRM (480p)      | [Link](https://huggingface.co/Skywork/Matrix-3D) |

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

        *   If using your own panorama image, structure it as:

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

        *   Set `--resolution` to `480` or `720`.  Generating a 720p video takes about an hour on an A800 GPU. Use `VISIBLE_GPU_NUM` for multi-GPU processing.

    3.  <span id="lowvram">**Low VRAM Mode:**</span>

        To run video generation with limited VRAM:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --enable_vram_management
        ```

    4.  **Extract 3D Scene:**

        *   **Optimization-based (High Quality):**

            ```bash
             python code/panoramic_video_to_3DScene.py \
                --inout_dir="./output/example1" \
                --resolution=720
            ```

            The 3D scene (`.ply` format) will be saved in `output/example1/generated_3dgs_opt.ply`.

        *   **Feed-forward Reconstruction (Efficient):**

            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
            ```

            The 3D scene (`.ply` format) and rendered videos will be saved in `output/example2`.  Customize the video and pose paths for your own data.

## Create Your Own: Customizable Movement

| Movement Mode         | Trajectory                                 | Panoramic Video                                                                 | 3D Scene                                                                    |
| :--------------------- | :------------------------------------------ | :------------------------------------------------------------------------------ | :-------------------------------------------------------------------------- |
| S-curve Travel         | <img src="asset/movement/s.PNG" height="120" width="120">     | <img src="asset/movement/s.gif" height="150" width="300">                                     | <img src="asset/movement/s_3D.gif" height="150">                         |
| Forward on the Right   | <img src="asset/movement/forward.PNG" height="120" width="120">  | <img src="asset/movement/forward.gif" height="150" width="300">                                | <img src="asset/movement/forward_3D.gif" height="150">                    |

*   **Movement Modes:** Choose from `Straight Travel`, `S-curve Travel`, or `Forward on the Right` using the `--movement_mode` option in `code/panoramic_image_to_video.py`.

*   **Custom Trajectories:**  Provide your own camera trajectory in .json format:

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
      --json_path YOUR_TRAJECTORY_FILE.json
    ```

    Use `code/generate_example_camera.py` and the provided sample (`./data/test_cameras/test_cam_front.json`) to generate your camera trajectories.

## Gradio Demo

Run the Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

*   **GPU Configuration:**  `--max_gpus`:
    *   `1`: Supports text-video-3D generation (62 GB+ GPU memory recommended).
    *   `N (N≥2)`: Supports text-video-3D and image-video-3D workflows.

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

This project builds upon the following resources:

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