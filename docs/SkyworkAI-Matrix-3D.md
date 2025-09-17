# Matrix-3D: Generate and Explore Omnidirectional 3D Worlds

**Create immersive, explorable 3D worlds from text or images with Matrix-3D, leveraging panoramic representation for expansive and detailed scene generation.**

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>  
</div>

<div align="center">
  <a href="https://github.com/SkyworkAI/Matrix-3D">
    <img src="https://img.shields.io/badge/GitHub-SkyworkAI/Matrix--3D-blue?style=flat&logo=github" alt="GitHub Repo"/>
  </a>
  <a href="https://matrix-3d.github.io/">
    <img src="https://img.shields.io/badge/Project_Page-orange?style=flat" alt="Project Page"/>
  </a>
  <a href="https://huggingface.co/Skywork/Matrix-3D">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model"/>
  </a>
  <a href="https://arxiv.org/pdf/2508.08086">
    <img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Report"/>
  </a>
  <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version"/>
</div>

## Key Features

*   **Omnidirectional Exploration:** Generate 360-degree, explorable 3D scenes.
*   **Text & Image Input:** Create worlds from text prompts or input images.
*   **High Controllability:** Customize trajectories and easily extend functionalities.
*   **Strong Generalization:** Built on robust 3D data and video model priors.
*   **Speed-Quality Balance:** Choose between rapid or detailed 3D reconstruction methods.

## Recent Updates

*   **September 2, 2025:** 🚀 Released a 5B model with a low-VRAM mode (12GB VRAM requirement).
*   **August 29, 2025:** 🎉 Launched a Gradio demo for interactive exploration.
*   **August 25, 2025:** 🎉 Provided a script for running generation with 19GB VRAM.
*   **August 12, 2025:** 🎉 Code, technical report, and project page released!

## Examples

### Image-to-Scene Generation

| Image                                                                                                                              | Panoramic Video                                                                                               | 3D Scene                                                                                                                            |
| :--------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;">                                                             | <img src="asset/i2p/i2p_2.gif" height="150" width="300">                                                        | <img src="asset/i2p/i2p_2_3D.gif" height="150">                                                                                     |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;">                                                             | <img src="asset/i2p/i2p_1.gif" height="150" width="300">                                                        | <img src="asset/i2p/i2p_1_3D.gif" height="150">                                                                                     |

### Text-to-Scene Generation

| Text                                                                                                                                | Panoramic Video                                                                                               | 3D Scene                                                                                                                            |
| :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| <div style="background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div> A floating island with a waterfall </div> | <img src="asset/t2p/t2p_1.gif" height="150" width="300">                                                        | <img src="asset/t2p/t2p_1_3D.gif" height="150">                                                                                     |
| <div style="background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"> <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div> an impressionistic winter landscape </div> | <img src="asset/t2p/t2p_2.gif" height="150" width="300">                                                        | <img src="asset/t2p/t2p_2_3D.gif" height="150">                                                                                     |

**Explore related projects:**  Check out [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) for real-time interactive world models.

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
3.  **Install dependencies:**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

Download the necessary models from Hugging Face:

| Model Name          | Description                      | Download                                                     |
| :------------------ | :------------------------------- | :----------------------------------------------------------- |
| Text2PanoImage      | text2panoimage_lora.safetensors | [Link](https://huggingface.co/Skywork/Matrix-3D)               |
| PanoVideoGen-480p   | pano_video_gen_480p.ckpt         | [Link](https://huggingface.co/Skywork/Matrix-3D)               |
| PanoVideoGen-720p   | pano_video_gen_720p.bin          | [Link](https://huggingface.co/Skywork/Matrix-3D)               |
| PanoVideoGen-720p-5B | pano_video_gen_720p_5b.safetensors | [Link](https://huggingface.co/Skywork/Matrix-3D)               |
| PanoLRM-480p        | pano_lrm_480p.pt                 | [Link](https://huggingface.co/Skywork/Matrix-3D)               |

## GPU VRAM Requirements

| Model Name          | VRAM (approx.) | VRAM with low-vram mode |
| :------------------ | :------------- | :--------------------- |
| Text2PanoImage      | ~16GB          | -                      |
| PanoVideoGen-480p   | ~40GB          | ~15GB                  |
| PanoVideoGen-720p   | ~60GB          | ~19GB                  |
| PanoVideoGen-720p-5B | ~19GB          | ~12GB                  |
| PanoLRM-480p        | ~80GB          | -                      |

**Note:** PanoLRM reconstruction requires significant VRAM; the optimization-based alternative uses approximately 10GB.

## Usage

1.  **Download Checkpoints:**

    ```bash
    python code/download_checkpoints.py
    ```
2.  **One-command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

    Or, generate step by step:

3.  **Step 1: Text/Image to Panorama Image:**

    *   **Text Prompt:**

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

    *   Organize your custom image with prompt in `output/example1`:

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

    *   Adjust resolution with `--resolution=[480, 720]`.
    *   Run video generation with low-VRAM mode:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --enable_vram_management
        ```
    *   Run video generation with 5B model:

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720 \
          --use_5b_model
        ```
5.  **Step 3: Extract 3D Scene:**

    *   **Optimization-based (High Quality):**

        ```bash
        python code/panoramic_video_to_3DScene.py \
           --inout_dir="./output/example1" \
           --resolution=720
        ```
    *   **Feed-forward (Efficient):**

        ```bash
        python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
        ```

## Create Your Own

| Movement Mode          | Trajectory                      | Panoramic Video                                                                                               | 3D Scene                                                                                                                              |
| :--------------------- | :------------------------------ | :------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| S-curve Travel         | <img src="asset/movement/s.PNG"  height="120"  width="120"  >                     | <img src="asset/movement/s.gif" height="150" width="300">                                                        | <img src="asset/movement/s_3D.gif" height="150">                                                                                        |
| Forward on the Right   | <img src="asset/movement/forward.PNG"  height="120"  width="120"  >              | <img src="asset/movement/forward.gif" height="150" width="300">                                                        | <img src="asset/movement/forward_3D.gif" height="150">                                                                                        |

Configure movement mode in `code/panoramic_image_to_video.py` using `--movement_mode`.

*   Use custom camera trajectories by specifying `--json_path YOUR_TRAJECTORY_FILE.json`.
*   Generate your trajectory with `code/generate_example_camera.py`.

## Gradio Demo

Run the Gradio demo for interactive exploration:

```bash
python code/matrix.py --max_gpus=1
```

**GPU Configuration:**
-   `--max_gpus=1`: Text-video-3D generation (>=62GB GPU memory recommended).
-   `--max_gpus=N, N≥2`: Supports both text-video-3D and image-video-3D generation.

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

Built upon the following:

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