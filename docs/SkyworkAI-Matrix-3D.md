# Matrix-3D: Generate Immersive, Omnidirectional 3D Worlds from Text or Images

Matrix-3D revolutionizes 3D world creation, allowing users to explore expansive, explorable environments generated from text prompts or images.  

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">
    <a href="https://matrix-3d.github.io/"><img src="https://img.shields.io/badge/📄-Project_Page-orange" alt="Project Page"/></a>
    <a href="https://huggingface.co/Skywork/Matrix-3D"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat" alt="Hugging Face Model"/></a>
    <img src="https://img.shields.io/badge/version-v0.1.0-green" alt="Version"/>
    <a href="https://arxiv.org/pdf/2508.08086"><img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Technical report"/></a>
</div>

**[Visit the original repository on GitHub](https://github.com/SkyworkAI/Matrix-3D)**

## Key Features

*   **Expansive Scene Generation:** Create large-scale, 360-degree explorable scenes, going beyond the limitations of traditional methods.
*   **High Controllability:** Generate scenes from both text and image inputs, with customizable trajectories for personalized experiences.
*   **Robust Generalization:** Leverages self-developed 3D data and video model priors for generating diverse, high-quality 3D scenes.
*   **Speed-Quality Balance:** Choose between two panoramic 3D reconstruction methods for rapid or detailed 3D reconstruction.

## News

*   **Sep 02, 2025:** 🎉 Released a 5B model with low-VRAM mode (12GB VRAM required).
*   **Aug 29, 2025:** 🎉 Launched a [Gradio demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) for easy exploration.
*   **Aug 25, 2025:** 🎉 Provided a [script](#lowvram) for running 720p video generation with 19GB VRAM.
*   **Aug 12, 2025:** 🎉 Public release of code, technical report, and project page.

## Image-to-Scene Generation

| Image | Panoramic Video | 3D Scene |
| ----- | --------------- | -------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_2.gif" height="150" width="300"> | <img src="asset/i2p/i2p_2_3D.gif" height="150"> |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"> | <img src="asset/i2p/i2p_1.gif" height="150" width="300"> | <img src="asset/i2p/i2p_1_3D.gif" height="150"> |

## Text-to-Scene Generation

| Text | Panoramic Video | 3D Scene |
| --- | --------------- | -------- |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall | <img src="asset/t2p/t2p_1.gif" height="150" width="300"> | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape | <img src="asset/t2p/t2p_2.gif" height="150" width="300"> | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

**Related Project:** Explore Real-Time Interactive Long-Sequence World Models with [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## Installation

Tested on Linux systems with NVIDIA GPUs.

```bash
# Clone the repository
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D

# Create a new conda environment
conda create -n matrix3d python=3.10
conda activate matrix3d

# Install torch and torchvision (with GPU support, we use CUDA 12.4 Version)
pip install torch==2.7.0 torchvision==0.22.0

# Run installation script
chmod +x install.sh
./install.sh
```

## Pretrained Models

| Model Name | Description | Download |
| :---------: | :----------: |  :-: |
| Text2PanoImage | text2panoimage\_lora.safetensors | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-480p | pano\_video\_gen\_480p.ckpt | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p | pano\_video\_gen\_720p.bin | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p-5B | pano\_video\_gen\_720p\_5b.safetensors | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoLRM-480p | pano\_lrm\_480p.pt | [Link](https://huggingface.co/Skywork/Matrix-3D) |

## GPU VRAM Requirements

| Model Name | VRAM | VRAM with low-vram mode |
| :---------: | :----------: | :----------: |
| Text2PanoImage | ~16GB | - |
| PanoVideoGen-480p | ~40GB | ~15GB |
| PanoVideoGen-720p | ~60GB | ~19GB |
| PanoVideoGen-720p-5B | ~19GB | ~12GB |
| PanoLRM-480p | ~80GB | - |

**Note:** PanoLRM inference is optional and can be replaced with optimization-based reconstruction, which uses ~10GB VRAM.

## Usage

*   🔧 **Checkpoint Download**

    ```bash
    python code/download_checkpoints.py
    ```

*   🔥 **One-command 3D World Generation**

    ```bash
    ./generate.sh
    ```

    Or generate step-by-step:

    1.  🖼️ **Step 1: Text/Image to Panorama Image**

        *   From text:

            ```bash
            python code/panoramic_image_generation.py \
                --mode=t2p \
                --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
                --output_path="./output/example1"
            ```

        *   From image:

            ```bash
            python code/panoramic_image_generation.py \
                --mode=i2p \
                --input_image_path="./data/image1.jpg" \
                --output_path="./output/example1"
            ```

        Create a directory with the `pano_img.jpg` and `prompt.txt` for your image as follows, then proceed to step 2:

        ```
        ./output/example1
        └─ pano_img.jpg
        └─ prompt.txt
        ```

    2.  📹 **Step 2: Generate Panoramic Video**

        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
            --inout_dir="./output/example1"  \
            --resolution=720
        ```

        Adjust resolution [480, 720]. The video will save in `output/example1/pano_video.mp4`.  Use `VISIBLE_GPU_NUM` for multi-GPU acceleration.

        *   <span id="lowvram">**Low VRAM mode**</span>

            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --enable_vram_management
            ```

        *   <span id="5B">**5B model**</span>

            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --use_5b_model
            ```

    3.  🏡 **Step 3: Extract 3D Scene**

        *   Optimization-based:

            ```bash
            python code/panoramic_video_to_3DScene.py \
                --inout_dir="./output/example1" \
                --resolution=720
            ```

        *   Feed-forward:

            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
            ```

## 🎬 Create Your Own

| Movement Mode | Trajectory | Panoramic Video | 3D Scene |
| :------------ | :--------- | :-------------- | :------- |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>S-curve Travel | <img src="asset/movement/s.PNG" height="120" width="120"> | <img src="asset/movement/s.gif" height="150" width="300"> | <img src="asset/movement/s_3D.gif" height="150"> |
| <div style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>Forward on the Right | <img src="asset/movement/forward.PNG" height="120" width="120"> | <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

Configure movement with `--movement_mode` in `code/panoramic_image_to_video.py`.  Use your own camera trajectory in `.json` format:

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720 \
  --json_path YOUR_TRAJECTORY_FILE.json
```

Refer to `./data/test_cameras/test_cam_front.json` and `code/generate_example_camera.py`.

## 🖱️ Gradio Demo

Run: `python code/matrix.py --max_gpus=1`

GPU Configuration Notes:

*   Single GPU (`--max_gpus=1`): Text-video-3D (62GB+ memory recommended).
*   Multiple GPUs (`--max_gpus=N, N≥2`): Supports both text-video-3D and image-video-3D generation.

## 📚 Citation

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}
```

---

## 🤝 Acknowledgements

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