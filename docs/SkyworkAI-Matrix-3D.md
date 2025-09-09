# Matrix-3D: Generate and Explore Omnidirectional 3D Worlds

**Unleash your imagination and step into explorable, panoramic 3D worlds with Matrix-3D!** [[Original Repo](https://github.com/SkyworkAI/Matrix-3D)]

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">
  [![Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
  [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
  [![Version](https://img.shields.io/badge/version-v0.1.0-green)](https://github.com/SkyworkAI/Matrix-3D/releases/tag/v0.1.0)
  [![Technical Report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)
</div>

## Key Features

*   **360° Exploration:** Create expansive, explorable scenes that allow for complete omnidirectional views.
*   **Text & Image Input:** Generate 3D worlds from both text prompts and existing images, providing high control over scene generation.
*   **High-Quality Scenes:** Leverage self-developed 3D data and video model priors for diverse, high-quality scene generation.
*   **Speed & Detail:** Offers two panoramic 3D reconstruction methods to achieve a balance between speed and detailed 3D reconstruction.

## What's New

*   **[Sep 02, 2025]:** 🎉 Released a 5B model with a low-VRAM mode that only requires 12GB VRAM!
*   **[Aug 29, 2025]:** 🎉 Launched a [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) for easy interaction!
*   **[Aug 25, 2025]:** 🎉 Provided a [script](#lowvram) for running the generation process with 19GB VRAM!
*   **[Aug 12, 2025]:** 🎉 Publicly released the code, technical report, and project page of Matrix-3D!

## Examples

### Image-to-Scene Generation

| Image                                                                                                                                                            | Panoramic Video                                      | 3D Scene                                             |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- | :--------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;">                                                                                        | <img src="asset/i2p/i2p_2.gif" height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150">       |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;">                                                                                        | <img src="asset/i2p/i2p_1.gif" height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150">       |

### Text-to-Scene Generation

| Text                                                                                                                                                                                                                                                                                                                                                   | Panoramic Video                                      | 3D Scene                                             |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- | :--------------------------------------------------- |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif" height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150">       |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</div> | <img src="asset/t2p/t2p_2.gif" height="150" width="300"> | <img src="asset/t2p/t2p_2_3D.gif" height="150">       |

**Related Project:** Explore real-time interactive long-sequence world models with [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

**Prerequisites:** Linux system with NVIDIA GPU.

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
    pip install torch==2.7.0 torchvision==0.22.0
    ```

4.  **Run Installation Script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name            | Description                                                                   | Download                                                                  |
| :-------------------- | :---------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| Text2PanoImage        | Generates panoramic images from text.                                         | [Link](https://huggingface.co/Skywork/Matrix-3D)                         |
| PanoVideoGen-480p     | Generates 480p panoramic videos.                                              | [Link](https://huggingface.co/Skywork/Matrix-3D)                         |
| PanoVideoGen-720p     | Generates 720p panoramic videos.                                              | [Link](https://huggingface.co/Skywork/Matrix-3D)                         |
| PanoVideoGen-720p-5B  | Generates 720p panoramic videos using a 5B model.                                | [Link](https://huggingface.co/Skywork/Matrix-3D)                         |
| PanoLRM-480p          | Used for feed-forward 3D scene reconstruction.                               | [Link](https://huggingface.co/Skywork/Matrix-3D)                         |

### VRAM Requirements

| Model Name           | VRAM (Estimated) | VRAM with Low-VRAM Mode |
| :------------------- | :--------------- | :--------------------- |
| Text2PanoImage       | \~16 GB           | -                      |
| PanoVideoGen-480p    | \~40 GB           | \~15 GB                 |
| PanoVideoGen-720p    | \~60 GB           | \~19 GB                 |
| PanoVideoGen-720p-5B | \~19 GB           | \~12 GB                 |
| PanoLRM-480p         | \~80 GB           | -                      |

**Note:** PanoLRM inference is optional; optimization-based reconstruction (see below) can replace it, using about 10GB VRAM.

## Usage

1.  **Download Checkpoints:**

    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

    Or follow these steps:

3.  **Step 1: Text/Image to Panorama Image**

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

        Place your input panorama image in `output/example1/pano_img.jpg` with a corresponding prompt file `output/example1/prompt.txt`.

4.  **Step 2: Generate Panoramic Video**

    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
    ```

    Use `--resolution` (480 or 720) to change video resolution. Generation may take about an hour on an A800 GPU. Use `VISIBLE_GPU_NUM` for multi-GPU acceleration.

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

5.  **Step 3: Extract 3D Scene**

    *   **Optimization-based reconstruction:**

        ```bash
         python code/panoramic_video_to_3DScene.py \
            --inout_dir="./output/example1" \
            --resolution=720
        ```

    *   **Feed-forward reconstruction:**

        ```bash
        python code/panoramic_video_480p_to_3DScene_lrm.py \
        --video_path="./data/case1/sample_video.mp4" \
        --pose_path='./data/case1/sample_cam.json' \
        --out_path='./output/example2'
        ```

## Create Your Own

| Movement Mode      | Trajectory                                 | Panoramic Video                          | 3D Scene                                     |
| :----------------- | :----------------------------------------- | :--------------------------------------- | :------------------------------------------- |
| S-curve Travel     | <img src="asset/movement/s.PNG" height="120" width="120">     | <img src="asset/movement/s.gif" height="150" width="300">      | <img src="asset/movement/s_3D.gif" height="150">          |
| Forward on the Right | <img src="asset/movement/forward.PNG" height="120" width="120"> | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

Available movement modes: `Straight Travel`, `S-curve Travel`, `Forward on the Right`. Configurable in `--movement_mode` in `code/panoramic_image_to_video.py`.  Use your own camera trajectories in `.json` format.

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

Use `code/generate_example_camera.py` to generate your camera trajectory.

## Gradio Demo

Launch the Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

GPU configuration notes:

*   `--max_gpus=1`: Text-video-3D generation (62 GB+ memory recommended).
*   `--max_gpus=N, N≥2`: Supports text-video-3D and image-video-3D (adjust GPU allocation).

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

Built on top of these projects, please cite them if you use them:
*   [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
*   [Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [WorldGen](https://github.com/ZiYang-xie/WorldGen/)
*   [MoGe](https://github.com/microsoft/MoGe)
*   [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
*   [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
*   [StableSR](https://github.com/IceClear/StableSR)
*   [VEnhancer](https://github.com/Vchitect/VEnhancer)

## Contact

For questions or feature requests, please open an issue.