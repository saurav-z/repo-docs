<h1 align="center">
    Matrix-3D: Generate and Explore Omnidirectional 3D Worlds
</h1>

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">
  <a href="https://matrix-3d.github.io/"><img src="https://img.shields.io/badge/📄-Project_Page-orange" alt="Project Page"></a>
  <a href="https://huggingface.co/Skywork/Matrix-3D"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model"></a>
  <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version">
  <a href="https://arxiv.org/pdf/2508.08086"><img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Report"></a>
</div>

**Unleash your imagination and create immersive, explorable 3D worlds from text or images with Matrix-3D!**

## Key Features

*   ✨ **Omnidirectional Exploration:** Generate complete 360-degree, explorable 3D scenes.
*   📝 **Text & Image Control:** Create worlds from text prompts or image inputs, offering high customization.
*   🚀 **Diverse Generation:** Built on strong 3D data and video model priors for high-quality and varied scene generation.
*   ⚖️ **Speed-Quality Balance:** Choose between two panoramic 3D reconstruction methods for rapid or detailed results.
*   🌐 **Large-Scale Scenes:** Generate expansive scenes beyond the limitations of traditional scene generation approaches.

## Image-to-Scene Generation Examples

| Image | Panoramic Video | 3D Scene |
|---|---|---|
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_2.gif" height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150"> |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_1.gif" height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150"> |

## Text-to-Scene Generation Examples

| Text Prompt | Panoramic Video | 3D Scene |
|---|---|---|
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">  <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</div> | <img src="asset/t2p/t2p_1.gif" height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">  <div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</div> | <img src="asset/t2p/t2p_2.gif" height="150" width="300"> | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

## Getting Started

*   **Installation:** Follow the instructions to set up your environment.
*   **Pretrained Models:** Download the necessary checkpoints.
*   **Usage:** Utilize the provided scripts to generate 3D worlds, step-by-step.
*   **Gradio Demo:** Launch a Gradio demo for interactive exploration.

For detailed instructions, please visit our [project repository](https://github.com/SkyworkAI/Matrix-3D).

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

3.  **Install Dependencies (CUDA 12.4 Recommended):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```
4.  **Run Installation Script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## Pretrained Models

| Model Name | Description | Download Link |
|---|---|---|
| Text2PanoImage | Generates panoramic images from text prompts | [Hugging Face](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-480p | Generates 480p panoramic videos | [Hugging Face](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p | Generates 720p panoramic videos | [Hugging Face](https://huggingface.co/Skywork/Matrix-3D) |
| PanoLRM-480p | Panoramic LRM model for fast 3D scene reconstruction | [Hugging Face](https://huggingface.co/Skywork/Matrix-3D) |

## Usage

1.  **Download Checkpoints:**

    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

3.  **Step-by-Step Generation (Alternative):**

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

    *   **Step 2: Generate Panoramic Video:**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```

        *   **Low VRAM Mode:** Run video generation with limited VRAM:
            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --enable_vram_management
            ```

    *   **Step 3: Extract 3D Scene:**

        *   Optimization-based Reconstruction:
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

### Movement Modes

We offer options for customizable camera trajectories.

| Movement Mode | Trajectory  | Panoramic Video  | 3D Scene |
|---|---|---|---|
|  S-curve Travel | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
|  Forward on the Right  | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

*   Use `--movement_mode` in `code/panoramic_image_to_video.py` to set the desired mode.
*   You can also supply your own camera trajectory in `.json` format.

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

## Gradio Demo

Launch a Gradio demo for easy interaction:

```bash
python code/matrix.py --max_gpus=1
```

*   **Single GPU:** Text-to-video-to-3D generation.  Requires 62GB of memory.
*   **Multiple GPUs:** Supports text-to-video-to-3D and image-to-video-to-3D workflows.

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

This project leverages the following:

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