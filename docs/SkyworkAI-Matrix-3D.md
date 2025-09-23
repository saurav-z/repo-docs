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

## Transform imagination into immersive experiences with Matrix-3D, generating explorable 3D worlds from text or images!  [Explore the original repo](https://github.com/SkyworkAI/Matrix-3D)

**Matrix-3D** is an innovative system for generating and exploring 3D worlds. Leveraging panoramic representations, it enables immersive, 360-degree exploration of scenes created from text prompts or images.

**Key Features:**

*   ✅ **Omnidirectional Exploration:** Generate complete 360-degree explorable scenes.
*   ✅ **Versatile Input:** Accepts both text and image inputs for scene generation.
*   ✅ **Customizable & Extensible:** Supports custom trajectories and infinite scene possibilities.
*   ✅ **High-Quality & Diverse Scenes:** Utilizes self-developed 3D data and video model priors.
*   ✅ **Speed-Quality Balance:** Offers two panoramic 3D reconstruction methods for rapid and detailed results.

## 📰 Recent Updates
*   **Sep 02, 2025:** 🎉 Released a 5B model with a low-VRAM mode for 12G VRAM compatibility!
*   **Aug 29, 2025:** 🎉 Launched a [Gradio Demo](https://github.com/SkyworkAI/Matrix-3D/tree/main?tab=readme-ov-file#%EF%B8%8F-gradio-demo) for easy visualization.
*   **Aug 25, 2025:** 🎉 Provided a [script](#lowvram) for 19G VRAM video generation.
*   **Aug 12, 2025:** 🎉 Project release: code, technical report, and project page available.

## Image-to-Scene Generation
| Image                                  | Panoramic Video                      | 3D Scene                         |
| :------------------------------------- | :----------------------------------- | :------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;">   | <img src="asset/i2p/i2p_2.gif"  height="150" width="300">   | <img src="asset/i2p/i2p_2_3D.gif" height="150">   |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;">   | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">   | <img src="asset/i2p/i2p_1_3D.gif" height="150">   |

## Text-to-Scene Generation
| Text                                       | Panoramic Video                      | 3D Scene                         |
| :----------------------------------------- | :----------------------------------- | :------------------------------- |
| <span style="background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3);">A floating island with a waterfall</span> | <img src="asset/t2p/t2p_1.gif"  height="150" width="300">   | <img src="asset/t2p/t2p_1_3D.gif" height="150">   |
| <span style="background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3);">an impressionistic winter landscape</span> | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" > | <img src="asset/t2p/t2p_2_3D.gif" height="150">   |

**Related Project**: Explore Real-Time Interactive Long-Sequence World Models at [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2).

## ⚙️ Installation
*Tested on Linux systems with NVIDIA GPU.*

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
3.  **Install PyTorch (with GPU Support):**  *Uses CUDA 12.4*
    ```bash
    pip install torch==2.7.0 torchvision==0.22.0
    ```
4.  **Run Installation Script:**
    ```bash
    chmod +x install.sh
    ./install.sh
    ```

## 💾 Pretrained Models
| Model Name                | Description                               | Download                                                              |
| :------------------------ | :---------------------------------------- | :-------------------------------------------------------------------- |
| Text2PanoImage          |  -                                          | [Link](https://huggingface.co/Skywork/Matrix-3D)                      |
| PanoVideoGen-480p         |  -                                          | [Link](https://huggingface.co/Skywork/Matrix-3D)                      |
| PanoVideoGen-720p         |  -                                          | [Link](https://huggingface.co/Skywork/Matrix-3D)                      |
| PanoVideoGen-720p-5B      |  -                                          | [Link](https://huggingface.co/Skywork/Matrix-3D)                      |
| PanoLRM-480p              |  -                                          | [Link](https://huggingface.co/Skywork/Matrix-3D)                      |

## 💡 GPU VRAM Requirements
The minimum GPU VRAM requirement to run the pipeline is **16GB**.

| Model Name          | VRAM (approximate) | VRAM with low-vram mode |
| :------------------ | :----------------- | :----------------------- |
| Text2PanoImage    | ~16GB              | -                        |
| PanoVideoGen-480p   | ~40GB              | ~15GB                    |
| PanoVideoGen-720p   | ~60GB              | ~19GB                    |
| PanoVideoGen-720p-5B| ~19GB              | ~12GB                    |
| PanoLRM-480p        | ~80GB              | -                        |

**Note:** The inference of PanoLRM is optional; optimization-based reconstruction (see below) requires about 10GB VRAM.

## 🚀 Usage

1.  **Download Checkpoints:**
    ```bash
    python code/download_checkpoints.py
    ```
2.  **One-Command 3D World Generation:**
    ```bash
    ./generate.sh
    ```
3.  **Step-by-Step Generation:**

    *   **Step 1: Text/Image to Panorama Image**

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
            Organize your own panorama image like this:
            ```
            ./output/example1
            └─ pano_img.jpg
            └─ prompt.txt
            ```

    *   **Step 2: Generate Panoramic Video**
        ```bash
        VISIBLE_GPU_NUM=1
        torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
          --inout_dir="./output/example1"  \
          --resolution=720
        ```

        *   **Low VRAM Mode:** (for 19GB VRAM)
            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --enable_vram_management
            ```

        *   **5B Model:** (fast and low-VRAM)
            ```bash
            VISIBLE_GPU_NUM=1
            torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
              --inout_dir="./output/example1"  \
              --resolution=720 \
              --use_5b_model
            ```

    *   **Step 3: Extract 3D Scene**

        *   Optimization-based Reconstruction:
            ```bash
             python code/panoramic_video_to_3DScene.py \
                --inout_dir="./output/example1" \
                --resolution=720
            ```
        *   Feed-forward Reconstruction:
            ```bash
            python code/panoramic_video_480p_to_3DScene_lrm.py \
            --video_path="./data/case1/sample_video.mp4" \
            --pose_path='./data/case1/sample_cam.json' \
            --out_path='./output/example2'
            ```

## 🎬 Create Your Own
| Movement Mode           | Trajectory          | Panoramic Video            | 3D Scene               |
| :---------------------- | :------------------ | :------------------------- | :--------------------- |
| S-curve Travel          | <img src="asset/movement/s.PNG"  height="120"  width="120"  > | <img src="asset/movement/s.gif" height="150"  width="300"> | <img src="asset/movement/s_3D.gif" height="150" > |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" >| <img src="asset/movement/forward.gif" height="150" width="300"> | <img src="asset/movement/forward_3D.gif" height="150"> |

*   **Movement Modes:**  `Straight Travel`, `S-curve Travel`, `Forward on the Right` (configure with `--movement_mode` in `code/panoramic_image_to_video.py`)
*   **Custom Trajectories:** Provide your own camera trajectory in .json format.
    ```bash
    VISIBLE_GPU_NUM=1
    torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
      --inout_dir="./output/example1"  \
      --resolution=720
      --json_path YOUR_TRAJECTORY_FILE.json
    ```
    Use `code/generate_example_camera.py` to generate your own camera trajectory.

## 🖱️ Gradio Demo

Run the following command to launch the Gradio demo:
```bash
python code/matrix.py --max_gpus=1
```
*   **GPU Configuration:**
    *   **Single GPU** (--max\_gpus=1): Supports text-video-3D generation. Requires at least 62 GB of memory.
    *   **Multiple GPUs** (--max\_gpus=N, N≥2): Supports text-video-3D and image-video-3D workflows.

## 📚 Citation
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