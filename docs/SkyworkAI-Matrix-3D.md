# Matrix-3D: Create and Explore Omnidirectional 3D Worlds

**Generate immersive, explorable 3D worlds from text or images with Matrix-3D, pushing the boundaries of 3D scene creation.**

[<img src="./asset/logo.PNG" alt="Matrix-3D Logo" width="400" style="margin-bottom: 5px;"/>](https://github.com/SkyworkAI/Matrix-3D)

[![Project Page](https://img.shields.io/badge/📄-Project_Page-orange)](https://matrix-3d.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=flat)](https://huggingface.co/Skywork/Matrix-3D)
[![Version](https://img.shields.io/badge/version-v0.1.0-green)](https://github.com/SkyworkAI/Matrix-3D)
[![Technical Report](https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2508.08086)

Matrix-3D is a cutting-edge project designed to generate and explore 3D worlds from text or images. It leverages panoramic representation for wide-coverage, omnidirectional 3D scene creation.

## Key Features

*   **Omnidirectional Exploration:** Generate expansive 360-degree explorable scenes.
*   **Versatile Input:** Supports both text and image inputs for scene creation.
*   **High Controllability:** Offers customizable camera trajectories and infinite extensibility.
*   **Strong Generalization:** Built upon self-developed 3D data and video model priors, ensuring diverse, high-quality scenes.
*   **Speed-Quality Balance:** Two panoramic 3D reconstruction methods: a fast and efficient method and a more detailed, high-quality approach.
*   **Low VRAM Mode:** Supports video generation with limited VRAM requirements (as low as 12GB).
*   **5B Model:** Provides a lighter-weight 5B model for faster video generation and lower VRAM usage.

## Quick Start

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
    cd Matrix-3D
    ```

2.  **Create and Activate a Conda Environment:**

    ```bash
    conda create -n matrix3d python=3.10
    conda activate matrix3d
    ```

3.  **Install Dependencies:**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0  # Replace with appropriate CUDA version
    chmod +x install.sh
    ./install.sh
    ```

###  Get Started
1.  **Download Checkpoints**:
    ```bash
    python code/download_checkpoints.py
    ```

2.  **One-Command 3D World Generation**
```bash
./generate.sh
```

### Usage Guide
Detailed instructions are given in the original readme

## Generated Examples

**Image-to-Scene Generation:**

| Image                                                                                                                                                              | Panoramic Video                                                                  | 3D Scene                                                                     |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| <img src="asset/i2p/i2p_2.png" style="width: 200px; border-radius: 6px;"><br> | <img src="asset/i2p/i2p_2.gif"  height="150" width="300">  | <img src="asset/i2p/i2p_2_3D.gif" height="150"> |
| <img src="asset/i2p/i2p_1.png" style="width: 200px; border-radius: 6px;"><br> | <img src="asset/i2p/i2p_1.gif"  height="150" width="300">  | <img src="asset/i2p/i2p_1_3D.gif" height="150"> |

**Text-to-Scene Generation:**

| Text                                                                                                                                                                                                                                                                                                     | Panoramic Video                                                                  | 3D Scene                                                                     |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| <span style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>A floating island with a waterfall</span> | <img src="asset/t2p/t2p_1.gif"  height="150" width="300">  | <img src="asset/t2p/t2p_1_3D.gif" height="150"> |
| <span style="font-family: 'Palatino', 'Georgia', serif; font-size: 1.3em; color: transparent; background: linear-gradient(45deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%); -webkit-background-clip: text; background-clip: text; text-shadow: 0 0 5px rgba(168,237,234,0.3), 0 0 10px rgba(254,214,227,0.3); padding: 15px; border: 1px solid rgba(168,237,234,0.5); border-radius: 8px; background-color: rgba(10,20,30,0.7); position: relative; overflow: hidden;"><div style="position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px; background: radial-gradient(circle at 20% 30%, rgba(254,214,227,0.1) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(168,237,234,0.1) 0%, transparent 40%); z-index: -1;"></div>an impressionistic winter landscape</span> | <img src="asset/t2p/t2p_2.gif"  height="150"  width="300" > | <img src="asset/t2p/t2p_2_3D.gif" height="150"> |

## GPU Memory Requirements

| Model Name | VRAM (Approximate) | VRAM with Low-VRAM Mode |
| :---------: | :----------: | :----------: |
| Text2PanoImage| ~16g | - |
| PanoVideoGen-480p| ~40g | ~15g |
| PanoVideoGen-720p| ~60g | ~19g |
| PanoVideoGen-720p-5B| ~19g | ~12g |
|PanoLRM-480p| ~80g | - |

**Note:**  The inference of PanoLRM will take lots of VRAM, but it is optional, you can replace it with the optimization-based reconstruction, which only takes about 10G VRAM.

## Create Your Own 3D Scene

You can customize the camera trajectory and movement mode. The following example shows the different movement modes

| Movement Mode          | Trajectory                           | Panoramic Video                                    | 3D Scene                                           |
| :--------------------- | :----------------------------------- | :------------------------------------------------- | :------------------------------------------------- |
| S-curve Travel          | <img src="asset/movement/s.PNG"  height="120"  width="120"  >           | <img src="asset/movement/s.gif" height="150"  width="300">             | <img src="asset/movement/s_3D.gif" height="150" >            |
| Forward on the Right | <img src="asset/movement/forward.PNG"  height="120"  width="120" >      | <img src="asset/movement/forward.gif" height="150" width="300">           | <img src="asset/movement/forward_3D.gif" height="150">           |


To provide your own camera trajectory, use a .json file in the specified format.  See `./data/test_cameras/test_cam_front.json` for an example.

## Gradio Demo

Launch the Gradio demo for a user-friendly interface:

```bash
python code/matrix.py --max_gpus=1
```

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