# LAM: One-Shot Animatable Gaussian Head - Official PyTorch Implementation

**Create stunning, interactive 3D avatars from a single image in seconds with LAM, the Large Avatar Model.**  This repository provides the official PyTorch implementation for LAM, a groundbreaking approach to 3D avatar generation and animation.  Explore the cutting-edge technology behind LAM and its capabilities for creating realistic and engaging digital humans.  For the original source code, visit the [original LAM repository](https://github.com/aigc3d/LAM).

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

## Key Features

*   **Blazing-Fast Avatar Generation:** Generate high-fidelity 3D avatars from a single image in seconds.
*   **Cross-Platform Compatibility:** Animate and render your avatars on any device, ensuring broad accessibility.
*   **Real-time Interactive Experiences:** Leverage the low-latency SDK for engaging interactive chatting avatars.

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

## What's New

*   **September 9, 2025:** PanoLAM Technical Report released - [Read More](https://arxiv.org/pdf/2509.07552)!
*   **May 20, 2025:** WebGL-Render Released - [Explore Here](https://github.com/aigc3d/LAM_WebRender)!
*   **May 10, 2025:**  Direct Avatar Export for OpenAvatarChat via ModelScope Demo - [Try it Now](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)!
*   **April 30, 2025:** Avatar Export Feature Released - [Get Started](tools/AVATAR_EXPORT_GUIDE.md) to use your LAM avatars in OpenAvatarChat!
*   **April 21, 2025:** WebGL Interactive Chatting Avatar SDK Launched on OpenAvatarChat - [Experience it Here](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)!
*   **April 19, 2025:** Audio2Expression Model Released - Animate your LAM avatars with audio! [Learn More](https://github.com/aigc3d/LAM_Audio2Expression)

## Get Started

### Online Demos

*   **Hugging Face Space:** Generate avatars from one image.  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
*   **ModelScope Space:**  Generate avatars from one image. [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
*   **OpenAvatarChat:** Experience interactive chatting avatars. [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat) &nbsp;  [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

**One-Click Installation (Windows):** A simplified installation package is available for Windows (Cuda 12.8), supported by "十字鱼".
*   **Video Guide:** [Watch the Tutorial](https://www.bilibili.com/video/BV13QGizqEey)
*   **Download:** [Get the Package](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)

#### Linux:

```bash
git clone https://github.com/aigc3d/LAM.git
cd LAM
# Install with Cuda 12.1
sh ./scripts/install/install_cu121.sh
# Or Install with Cuda 11.8
sh ./scripts/install/install_cu118.sh
```

#### Windows:

For Windows, please refer to the [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

### Model Weights

| Model        | Training Data           | HuggingFace                                                                             | ModelScope                                                                                                                                 | Reconstruction Time | A100 (A & R)     | XiaoMi 14 Phone (A & R) |
|--------------|-------------------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|---------------------|------------------|-------------------------|
| LAM-20K      | VFHQ                    | TBD                                                                                     | TBD                                                                                                                                         | 1.4 s               | 562.9FPS         | 110+FPS                 |
| LAM-20K      | VFHQ + NeRSemble        | [Link](https://huggingface.co/3DAIGC/LAM-20K)                                           | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)                                                                           | 1.4 s               | 562.9FPS         | 110+FPS                 |
| LAM-20K      | Our large dataset       | TBD                                                                                     | TBD                                                                                                                                         | 1.4 s               | 562.9FPS         | 110+FPS                 |

(**A & R:** Animating & Rendering)

#### HuggingFace Download

```bash
# Download Assets
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

#### ModelScope Download

```bash
pip3 install modelscope
# Download Assets
modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
modelscope download "Damo_XR_Lab/LAM-20K" --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
```

### Gradio Run

```bash
python app_lam.py
```

To export ZIP files for real-time conversations on OpenAvatarChat, see the [Guide](tools/AVATAR_EXPORT_GUIDE.md).

```bash
python app_lam.py --blender_path /path/blender
```

### Inference

```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

### Acknowledgements

This project builds upon the work of many researchers and open-source projects, including:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

We are grateful for their valuable contributions.

### Explore More

Check out our other related projects:
*   [LHM](https://github.com/aigc3d/LHM)

### Citation

```
@inproceedings{he2025lam,
  title={LAM: Large Avatar Model for One-shot Animatable Gaussian Head},
  author={He, Yisheng and Gu, Xiaodong and Ye, Xiaodan and Xu, Chao and Zhao, Zhengyi and Dong, Yuan and Yuan, Weihao and Dong, Zilong and Bo, Liefeng},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages={1--13},
  year={2025}
}
```