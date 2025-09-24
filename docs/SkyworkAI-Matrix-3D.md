<h1 align="center">
    Matrix-3D: Generate and Explore Omnidirectional 3D Worlds
</h1>

<div align="center">
  <img src="./asset/logo.PNG" alt="logo" width="800" style="margin-bottom: 5px;"/>
</div>

<div align="center">

[![📄 Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
![Badge](https://img.shields.io/badge/version-v0.1.0-green)
[![Technical report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)
</div>

## ✨ **Transform your imagination into explorable 3D worlds with Matrix-3D!**  [Back to Top](https://github.com/SkyworkAI/Matrix-3D)

Matrix-3D offers a powerful and versatile solution for generating immersive, 360-degree explorable 3D environments from text or images, combining conditional video generation with advanced 3D reconstruction techniques.

**Key Features:**

*   **🌎 Large-Scale Scene Generation:** Create expansive, 360-degree explorable scenes, going beyond the limitations of traditional approaches.
*   **🕹️ High Controllability:**  Fine-tune your creations with both text and image inputs, customizable camera trajectories, and infinite scene extensibility.
*   **💡 Strong Generalization:** Leverage self-developed 3D data and video model priors for diverse, high-quality 3D scene generation.
*   **⚡ Speed-Quality Balance:** Choose between two panoramic 3D reconstruction methods for rapid or detailed 3D scene creation.

## 📰 What's New

*   **September 2, 2025:** 🎉 Introducing a 5B model with low-VRAM mode, requiring only 12GB VRAM!
*   **August 29, 2025:** 🎉 Interactive exploration now available with a [Gradio demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo).
*   **August 25, 2025:** 🎉  Run 720p generation with just 19GB of VRAM using our [low VRAM script](#lowvram).
*   **August 12, 2025:** 🎉 Matrix-3D code, technical report, and project page released!

## 🖼️ Image-to-Scene Generation

| Image                                                                                                                                  | Panoramic Video                                                                                                | 3D Scene                                                                                                    |
| :------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br>                                                         | <img src="asset/i2p/i2p_2.gif" height="150" width="300">                                                       | <img src="asset/i2p/i2p_2_3D.gif" height="150">                                                               |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br>                                                         | <img src="asset/i2p/i2p_1.gif" height="150" width="300">                                                       | <img src="asset/i2p/i2p_1_3D.gif" height="150">                                                               |

## ✍️ Text-to-Scene Generation

| Text                                                                                                                                                                                                                                                                                                                                  | Panoramic Video                                                                                                | 3D Scene                                                                                                    |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| <div style=" font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">  <div style="  position: absolute;  top: -10px;  left: -10px;  right: -10px;  bottom: -10px;  background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall </div> | <img src="asset/t2p/t2p_1.gif" height="150" width="300">                                                       | <img src="asset/t2p/t2p_1_3D.gif" height="150">                                                               |
| <div style=" font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;">  <div style="  position: absolute;  top: -10px;  left: -10px;  right: -10px;  bottom: -10px;  background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape </div> | <img src="asset/t2p/t2p_2.gif" height="150" width="300" >                                                       | <img src="asset/t2p/t2p_2_3D.gif" height="150">                                                               |

**Related Projects:** Explore Real-Time Interactive Long-Sequence World Models with [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## ⚙️ Installation

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

3.  **Install PyTorch (with CUDA support - tested with CUDA 12.4):**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```

4.  **Run Installation Script:**

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## 💾 Pretrained Models

Download the required models from Hugging Face:

| Model Name            | Description                                   | Download                                                                 |
| :-------------------: | :--------------------------------------------: | :-----------------------------------------------------------------------: |
| Text2PanoImage      | Text-to-Panorama Image Generation              | [Link](https://huggingface.co/Skywork/Matrix-3D)                        |
| PanoVideoGen-480p     | Panoramic Video Generation (480p)            | [Link](https://huggingface.co/Skywork/Matrix-3D)                        |
| PanoVideoGen-720p     | Panoramic Video Generation (720p)            | [Link](https://huggingface.co/Skywork/Matrix-3D)                        |
| PanoVideoGen-720p-5B  | Panoramic Video Generation (720p) - 5B Model   | [Link](https://huggingface.co/Skywork/Matrix-3D)                        |
| PanoLRM-480p          | Panoramic 3D Scene Reconstruction (480p)      | [Link](https://huggingface.co/Skywork/Matrix-3D)                        |

## 🖥️ GPU VRAM Requirements

The minimum VRAM requirement is 16GB.

| Model Name            | VRAM Usage | VRAM with Low-VRAM Mode |
| :-------------------: | :--------: | :---------------------: |
| Text2PanoImage      |   ~16GB    |           -             |
| PanoVideoGen-480p     |   ~40GB    |         ~15GB           |
| PanoVideoGen-720p     |   ~60GB    |         ~19GB           |
| PanoVideoGen-720p-5B  |   ~19GB    |         ~12GB           |
| PanoLRM-480p          |   ~80GB    |           -             |

**Note:** PanoLRM reconstruction is optional; an optimization-based reconstruction (see below) can be used instead and requires only about 10GB of VRAM.

## 🚀 Usage

1.  **Download Checkpoints:**

    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation:**

    ```bash
    ./generate.sh
    ```

    Or, generate step-by-step:

3.  **Step 1: Text/Image to Panorama Image:**

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

        Organize your panorama image with its prompt as following structure if using your own or results from other methods:

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

    Set `--resolution` to `480` or `720`.  Generating a 720p video typically takes about an hour on an A800 GPU; speed up with multi-GPU with  `VISIBLE_GPU_NUM`.

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

5.  **Step 3: Extract 3D Scene:**

    *   **Optimization-based Reconstruction:**

        ```bash
         python code/panoramic_video_to_3DScene.py \
            --inout_dir="./output/example1" \
            --resolution=720
        ```

    *   **Feed-Forward Reconstruction:**

        ```bash
        python code/panoramic_video_480p_to_3DScene_lrm.py \
        --video_path="./data/case1/sample_video.mp4" \
        --pose_path='./data/case1/sample_cam.json' \
        --out_path='./output/example2'
        ```

## 🎬 Create Your Own

| Movement Mode          | Trajectory                                                                     | Panoramic Video                                                                                              | 3D Scene                                                                                                     |
| :--------------------- | :----------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| S-curve Travel         | <img src="asset/movement/s.PNG"  height="120"  width="120"  >                    | <img src="asset/movement/s.gif" height="150"  width="300">                                                       | <img src="asset/movement/s_3D.gif" height="150" >                                                               |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" >                | <img src="asset/movement/forward.gif" height="150" width="300">                                                       | <img src="asset/movement/forward_3D.gif" height="150">                                                               |

Configure the `--movement_mode`  in `code/panoramic_image_to_video.py`:  Choose from  `Straight Travel`,  `S-curve Travel`, and  `Forward on the Right`.

You can also define your own camera trajectory in `.json` format, using world-to-camera matrices in OpenCV format.  Refer to `./data/test_cameras/test_cam_front.json` and use `code/generate_example_camera.py` to generate your own.

```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1"  \
  --resolution=720
  --json_path YOUR_TRAJECTORY_FILE.json
```

## 🖱️ Gradio Demo

Launch the Gradio demo:

```bash
python code/matrix.py --max_gpus=1
```

**GPU Configuration Notes:**

*   **Single GPU (`--max_gpus=1`):** Supports text-video-3D generation.  Requires at least 62GB of GPU memory for smooth operation.
*   **Multiple GPUs (`--max_gpus=N, N≥2`):**  Supports both text-video-3D and image-video-3D workflows. Optimize performance based on your hardware.

## 📚 Citation

If you find this project helpful, please cite it:

```bibtex
@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2508.08086},
  year      = {2025}
}

@article{dong2025panolora,
  title     = {PanoLora: Bridging Perspective and Panoramic Video Generation with LoRA Adaptation},
  author    = {Zeyu Dong and Yuyang Yin and Yuqi Li and Eric Li and Hao-Xiang Guo and Yikai Wang},
  journal   = {arXiv preprint arXiv:2509.11092},
  year      = {2025}
}
```

---

## 🤝 Acknowledgements

This project builds upon the following open-source projects:

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