# Matrix-3D: Generate and Explore Omnidirectional 3D Worlds

**Create immersive 3D worlds from text or images with Matrix-3D, enabling 360-degree exploration and unparalleled creative control.**  Check out the original repository for the full details: [Matrix-3D GitHub](https://github.com/SkyworkAI/Matrix-3D).

<div align="center">
  <img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">

[![📄 Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
![Badge](https://img.shields.io/badge/version-v0.1.0-green)
[![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)

</div>

## Key Features

*   **Omnidirectional Exploration:** Generate 3D scenes that allow for full 360-degree exploration.
*   **Text and Image Input:** Create worlds from both text prompts and images, providing versatile control.
*   **Customizable Trajectories:** Define your own camera paths for unique exploration experiences.
*   **High-Quality & Diverse Scenes:** Built on advanced 3D data and video model priors, generating stunning results.
*   **Speed-Quality Balance:** Choose between rapid and detailed 3D reconstruction methods.

## News

*   **August 29, 2025:** 🎉 Interactive [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) Available!
*   **August 25, 2025:** 🎉 Low-VRAM Script Released (19G VRAM)!
*   **August 12, 2025:** 🎉 Code, Technical Report, and Project Page Released!

## Examples

### Image-to-Scene Generation

| Image                                                                                                | Panoramic Video                                                                                               | 3D Scene                                                                                       |
| :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;">                          | <img src="asset/i2p/i2p_2.gif"  height="150" width="300">                                                     | <img src="asset/i2p/i2p_2_3D.gif" height="150">                                                |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;">                          | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">                                                     | <img src="asset/i2p/i2p_1_3D.gif" height="150">                                                |

### Text-to-Scene Generation

| Text                                                                                                                                                                                                                                                                                          | Panoramic Video                                                                                               | 3D Scene                                                                                       |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- |
| <th width="200" style="  font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;  "> <div style=" position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1; "></div>A floating island with a waterfall </th>                                                      | <img src="asset/t2p/t2p_1.gif"  height="150" width="300">                                                     | <img src="asset/t2p/t2p_1_3D.gif" height="150">                                                |
| <th width="200" style="  font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;  "> <div style=" position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1; "></div>an impressionistic winter landscape</th>                                                      | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" >                                                     | <img src="asset/t2p/t2p_2_3D.gif" height="150">                                                |

## Installation

*   Requires a Linux system with an NVIDIA GPU.
*   Clone the repository and set up your environment:

```bash
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D
conda create -n matrix3d python=3.10
conda activate matrix3d
pip install torch==2.7.0 torchvision==0.22.0  # CUDA 12.4
chmod +x install.sh
./install.sh
```

## Pretrained Models

| Model Name          | Description               | Download                                                                            |
| :------------------ | :------------------------ | :---------------------------------------------------------------------------------- |
| Text2PanoImage      |  -   | [Link](https://huggingface.co/Skywork/Matrix-3D)    |
| PanoVideoGen-480p   | -        | [Link](https://huggingface.co/Skywork/Matrix-3D) |
| PanoVideoGen-720p   | -          | [Link](https://huggingface.co/Skywork/Matrix-3D)   |
| PanoLRM-480p       |  -         | [Link](https://huggingface.co/Skywork/Matrix-3D)     |

## Usage

*   **Download Checkpoints:**

```bash
python code/download_checkpoints.py
```

*   **One-command 3D World Generation:**

```bash
./generate.sh
```

*   **Step-by-Step Generation:**

1.  **Image/Text to Panorama Image:**

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

2.  **Generate Panoramic Video:**

    ```bash
    VISIBLE_GPU_NUM=1 torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
    ```

    *   **Low VRAM Mode:**

    ```bash
    VISIBLE_GPU_NUM=1 torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720 \
      --enable_vram_management
    ```

3.  **Extract 3D Scene:**

    *   Optimization-Based:

    ```bash
     python code/panoramic_video_to_3DScene.py \
        --inout_dir="./output/example1" \
        --resolution=720
    ```

    *   Feed-Forward:

    ```bash
    python code/panoramic_video_480p_to_3DScene_lrm.py \
    --video_path="./data/case1/sample_video.mp4" \
    --pose_path='./data/case1/sample_cam.json' \
    --out_path='./output/example2'
    ```

## Create Your Own

| Movement Mode       | Trajectory                                         | Panoramic Video                                                                                               | 3D Scene                                                                                       |
| :------------------- | :------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- |
| S-curve Travel       | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300">                                                     | <img src="asset/movement/s_3D.gif" height="150" >                                                |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" >  | <img src="asset/movement/forward.gif" height="150" width="300">                                                     | <img src="asset/movement/forward_3D.gif" height="150">                                                |

*   Configure camera movement with `--movement_mode` in `code/panoramic_image_to_video.py`.
*   Use your own camera trajectories with the `--json_path` flag. See `./data/test_cameras/test_cam_front.json` for the format.
*   Generate your own camera trajectories using `code/generate_example_camera.py`.

## Gradio Demo

Launch the interactive Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

*   **GPU Configuration:**
    *   `--max_gpus=1`: Supports text-video-3D generation (62GB+ VRAM recommended).
    *   `--max_gpus=N (N≥2)`: Supports both text-video-3D and image-video-3D generation.  Allocate GPUs based on your hardware.

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

This project builds upon:
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