# Matrix-3D: Generate Interactive 3D Worlds from Text or Images

Matrix-3D revolutionizes 3D world creation by enabling the generation of explorable, omnidirectional 3D environments from simple text descriptions or images.

[![Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
![Badge](https://img.shields.io/badge/version-v0.1.0-green)
[![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)

**[Original Repo](https://github.com/SkyworkAI/Matrix-3D)**

## Key Features

*   ✨ **Omnidirectional Exploration:** Generate and explore 360-degree worlds.
*   🖼️ **Text-to-3D & Image-to-3D:** Convert text prompts or images into immersive 3D scenes.
*   🚀 **High Controllability:** Customize scenes with text and image inputs, trajectory control and infinite extensibility.
*   💡 **Strong Generalization:** Leverages pre-trained 3D data and video models for diverse, high-quality scenes.
*   ⚡ **Speed-Quality Balance:** Offers two 3D reconstruction methods: rapid and detailed.
*   💾 **Low VRAM Mode & 5B Model:** Optimized to run on systems with as little as 12GB of VRAM.

## Demo

### Image-to-Scene Generation

| Image                                                                                                                           | Panoramic Video                                                                          | 3D Scene                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;">                                                         | <img src="asset/i2p/i2p_2.gif"  height="150" width="300">                                | <img src="asset/i2p/i2p_2_3D.gif" height="150">                             |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;">                                                         | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">                                | <img src="asset/i2p/i2p_1_3D.gif" height="150">                             |

### Text-to-Scene Generation

| Text                                                                                                                                                                        | Panoramic Video                                                                          | 3D Scene                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| <div style=" font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif"  height="150" width="300">                                | <img src="asset/t2p/t2p_1_3D.gif" height="150">                             |
| <div style=" font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</div> | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" >                                | <img src="asset/t2p/t2p_2_3D.gif" height="150">                             |

## News

*   **Sep 02, 2025:** 🎉 Released a 5B model with low-VRAM mode (12G VRAM required).
*   **Aug 29, 2025:** 🎉 Launched a [Gradio demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo).
*   **Aug 25, 2025:** 🎉 Provided a [script](#lowvram) for 19G VRAM video generation.
*   **Aug 12, 2025:** 🎉 Released code, technical report, and project page.

## Installation

Tested on Linux with NVIDIA GPU.

1.  **Clone Repository:**

    ```bash
    git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
    cd Matrix-3D
    ```

2.  **Create Conda Environment:**

    ```bash
    conda create -n matrix3d python=3.10
    conda activate matrix3d
    ```

3.  **Install Dependencies:**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0  # CUDA 12.4
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name           | Description                                     | Download                                                            |
| -------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| Text2PanoImage       | Generates panoramic images from text.         | [Link](https://huggingface.co/Skywork/Matrix-3D)                     |
| PanoVideoGen-480p    | Generates panoramic videos at 480p resolution.  | [Link](https://huggingface.co/Skywork/Matrix-3D)                     |
| PanoVideoGen-720p    | Generates panoramic videos at 720p resolution.  | [Link](https://huggingface.co/Skywork/Matrix-3D)                     |
| PanoVideoGen-720p-5B | 5B model for faster and more efficient video generation. | [Link](https://huggingface.co/Skywork/Matrix-3D)                     |
| PanoLRM-480p         | Optimized 3D reconstruction from video.           | [Link](https://huggingface.co/Skywork/Matrix-3D)                     |

## GPU VRAM Requirements

| Model Name           | VRAM (Approximate) | Low-VRAM Mode |
| -------------------- | ------------------- | ------------- |
| Text2PanoImage       | ~16 GB             | -             |
| PanoVideoGen-480p    | ~40 GB             | ~15 GB        |
| PanoVideoGen-720p    | ~60 GB             | ~19 GB        |
| PanoVideoGen-720p-5B | ~19 GB             | ~12 GB        |
| PanoLRM-480p         | ~80 GB             | -             |

**Note:** PanoLRM requires significant VRAM; the optimization-based reconstruction (see below) is a more efficient alternative.

## Usage

1.  **Checkpoint Download:**
    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command Generation:**
    ```bash
    ./generate.sh
    ```

    **OR Step-by-Step:**

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
        *   **Using External Images**: Place your panorama image in `output/example1/pano_img.jpg` and create `output/example1/prompt.txt` containing the image prompt.

    *   **Step 2: Generate Panoramic Video:**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```

        Adjust `--resolution` to 480 or 720.

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

| Movement Mode        | Trajectory          | Panoramic Video                          | 3D Scene                             |
| -------------------- | ------------------- | ---------------------------------------- | ------------------------------------ |
| S-curve Travel       | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

*   Configure `--movement_mode` in `code/panoramic_image_to_video.py`.
*   Use a custom camera trajectory in .json format with `--json_path YOUR_TRAJECTORY_FILE.json`.

    *   Use `code/generate_example_camera.py` to create trajectories.

## Gradio Demo

Run: `python code/matrix.py --max_gpus=1`

*   **GPU Configuration:**

    *   Single GPU (`--max_gpus=1`): Supports text-video-3D generation (62GB+ VRAM recommended).
    *   Multiple GPUs (`--max_gpus=N, N≥2`): Supports both text-video-3D and image-video-3D workflows; optimize performance through GPU allocation.

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

*   FLUX.1
*   Wan2.1
*   WorldGen
*   MoGe
*   nvdiffrast
*   gaussian-splatting
*   StableSR
*   VEnhancer

## Contact

For questions or feature requests, please open an issue.