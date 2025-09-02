# Matrix-3D: Generate Interactive 3D Worlds from Text and Images

Matrix-3D empowers users to explore expansive, 360-degree 3D worlds generated from simple text descriptions or images. 

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="800" style="margin-bottom: 5px;"/>  
</div>

<div align="center">
    <a href="https://matrix-3d.github.io/">
        <img src="https://img.shields.io/badge/📄-Project_Page-orange" alt="Project Page">
    </a>
    <a href="https://huggingface.co/Skywork/Matrix-3D">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model">
    </a>
    <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version">
    <a href="https://arxiv.org/pdf/2508.08086">
        <img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Report">
    </a>
</div>

**[Explore the original repo](https://github.com/SkyworkAI/Matrix-3D) for the full code and details.**

## Key Features:

*   **Omnidirectional Exploration:** Generate complete 360-degree explorable scenes.
*   **Text & Image Input:** Create worlds from text prompts or input images.
*   **Customizable Trajectories:** Define your own camera paths for unique experiences.
*   **High-Quality & Diverse Scenes:** Leverage advanced 3D data and video model priors.
*   **Speed & Detail Options:** Choose between rapid or detailed 3D reconstruction methods.
*   **Low VRAM Mode:** Includes a script for running with 19G VRAM.

## Recent Updates:

*   **August 29, 2025:** [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) available for easy visualization.
*   **August 25, 2025:**  [Script](#lowvram) provided to run generation with limited VRAM (19G).
*   **August 12, 2025:** Code, technical report, and project page released!

## Visual Examples

### Image-to-Scene Generation

| Image                                                                                                                                                                                          | Panoramic Video                                                                                           | 3D Scene                                                                                      |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br>                                                                                                        | <img src="asset/i2p/i2p_2.gif" height="150" width="300">                                                 | <img src="asset/i2p/i2p_2_3D.gif" height="150">                                                    |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br> | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">                                         | <img src="asset/i2p/i2p_1_3D.gif" height="150">                                                      |

### Text-to-Scene Generation

| Text                                                                                                                                                                                              | Panoramic Video                                                                                                  | 3D Scene                                                                                             |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| <th width="200" style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</th> | <img src="asset/t2p/t2p_1.gif" height="150" width="300">                                                         | <img src="asset/t2p/t2p_1_3D.gif" height="150">                                                            |
| <th width="200" style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</th> | <img src="asset/t2p/t2p_2.gif" height="150"  width="300" >                                                         | <img src="asset/t2p/t2p_2_3D.gif" height="150">                                                             |

**Related Project:**  Explore real-time interactive world models with [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux with NVIDIA GPU.

```bash
# Clone the repository 
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D

# Create a new conda environment
conda create -n matrix3d python=3.10
conda activate matrix3d

# Install torch and torchvision (with GPU support, we use CUDA 12.4 Version)
pip install torch==2.7.0 torchvision==0.22.0

#Run installation script
chmod +x install.sh
./install.sh
```

## GPU VRAM Requirements

| Model Name                 | VRAM (Approximate) |
| :--------------------------: | :----------------: |
| PanoVideoGen-480p w.o. vram management | ~40g |
| PanoVideoGen-720p w.o. vram management | ~60g |
| PanoVideoGen-720p-5b w.o. vram management | ~19g |
| PanoVideoGen-480p w. vram management  | ~15g |
| PanoVideoGen-720p w. vram management  | ~19g |
| PanoVideoGen-720p-5b w. vram management | ~12g |

## Pretrained Models

Download models from Hugging Face:

| Model Name          | Description            | Download Link                                                                                                                      |
| :-------------------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| Text2PanoImage      | Text to Panoramic Image | [Link](https://huggingface.co/Skywork/Matrix-3D)                                                                                  |
| PanoVideoGen-480p   | 480p Video Generation | [Link](https://huggingface.co/Skywork/Matrix-3D)                                                                                  |
| PanoVideoGen-720p   | 720p Video Generation | [Link](https://huggingface.co/Skywork/Matrix-3D)                                                                                  |
| PanoVideoGen-720p-5b | 5B Video Generation   | [Link](https://huggingface.co/Skywork/Matrix-3D)                                                                                  |
| PanoLRM-480p        | 3D Scene Reconstruction| [Link](https://huggingface.co/Skywork/Matrix-3D)                                                                                  |

## Usage

*   **Download Checkpoints:**

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

            Organize your own panorama images with a `pano_img.jpg` and `prompt.txt` in a directory.

    2.  **Generate Panoramic Video:**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```

        Adjust `--resolution` to 480 or 720.  Increase `VISIBLE_GPU_NUM` for multi-GPU processing.

        *   <span id="lowvram">**Low VRAM Mode:**</span>

            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --enable_vram_management
            ```

        *   <span id="5b">**5b Model:**</span>

            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --use_5b_model
            ```

    3.  **Extract 3D Scene:**

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

## Create Your Own Scenes:

| Movement Mode          | Trajectory                                   | Panoramic Video                                                      | 3D Scene                                                        |
| :----------------------: | :-------------------------------------------: | :-------------------------------------------------------------------: | :--------------------------------------------------------------: |
| S-curve Travel             | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300">  | <img src="asset/movement/s_3D.gif" height="150">            |
| Forward on the Right   | <img src="asset/movement/forward.PNG"  height="120"  width="120" > | <img src="asset/movement/forward.gif" height="150" width="300">   | <img src="asset/movement/forward_3D.gif" height="150">   |

Use `--movement_mode` in `code/panoramic_image_to_video.py` to select from: `Straight Travel`, `S-curve Travel`, or `Forward on the Right`.  Provide custom camera trajectories in .json format via `--json_path YOUR_TRAJECTORY_FILE.json`.  Generate trajectories using `code/generate_example_camera.py` and review the sample file (`./data/test_cameras/test_cam_front.json`).

## Gradio Demo

Run the interactive demo:

```bash
python code/matrix.py --max_gpus=1
```

*   Single GPU (`--max_gpus=1`): Text-video-3D generation. Requires at least 62 GB memory.
*   Multiple GPUs (`--max_gpus=N, N≥2`): Supports text-video-3D and image-video-3D workflows.  Adjust GPU allocation as needed.

## Citation

If you find this project useful, please cite:

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

This project is built upon the following:

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