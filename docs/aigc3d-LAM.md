# LAM: Large Avatar Model - Create Realistic 3D Avatars from a Single Image

LAM revolutionizes 3D avatar creation, allowing you to generate animatable Gaussian heads from a single image in seconds, making interactive chatting a reality.  Explore the official PyTorch implementation and unlock the potential of lifelike digital humans! [Check out the original repository](https://github.com/aigc3d/LAM).

[![Website](https://img.shields.io/badge/🏠-Website-blue)](https://aigc3d.github.io/projects/LAM/) 
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2502--17796-green)](https://arxiv.org/pdf/2502.17796)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
[![ModelScope](https://img.shields.io/badge/🧱-ModelScope-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model) 
[![Apache License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

### SIGGRAPH 2025

#####  <p align="center"> Yisheng He*, Xiaodong Gu*, Xiaodan Ye, Chao Xu, Zhengyi Zhao, Yuan Dong†, Weihao Yuan†, Zilong Dong, Liefeng Bo </p>

#####  <p align="center"> Tongyi Lab, Alibaba Group</p>

<p align="center">
  <img src="./assets/images/teaser.jpg" width="100%">
</p>

## Key Features

*   **One-Shot Avatar Generation:** Create incredibly realistic 3D avatars from a single image in just seconds.
*   **Fast Animation & Rendering:** Experience rapid cross-platform animation and rendering on any device.
*   **Real-time Interaction:**  Benefit from a low-latency SDK designed for real-time interactive chatting avatars.

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

## What's New

Stay up-to-date with the latest LAM developments:

*   **[September 9, 2025]:** Released the technical report of [PanoLAM](https://arxiv.org/pdf/2509.07552)!
*   **[May 20, 2025]:** Released the [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)!
*   **[May 10, 2025]:** ModelScope Demo now supports exporting avatars for OpenAvatarChat.
*   **[April 30, 2025]:**  Avatar Export Feature released ([tools/AVATAR_EXPORT_GUIDE.md]) for chatting with LAM-generated avatars on OpenAvatarChat.
*   **[April 21, 2025]:**  Interactive Chatting Avatar SDK released on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat), including LLM, ASR, TTS, and Avatar capabilities.
*   **[April 19, 2025]:** Released the [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model for audio-driven avatar animation.

## Get Started

### Online Demos

Explore LAM's capabilities with these online demos:

*   **Hugging Face:**  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
*   **ModelScope:**  [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)

**Interactive Chatting:**

*   **Hugging Face:**  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
*   **ModelScope:**  [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

Choose your preferred setup method:

*   **Windows One-Click Installation:**  Simplify the setup with a pre-built package for Windows (Cuda 12.8).
    *   [Video](https://www.bilibili.com/video/BV13QGizqEey)
    *   [Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)

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

Refer to the [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

### Model Weights

| Model         | Training Data                  | HuggingFace | ModelScope | Reconstruction Time | A100 (A & R) |  XiaoMi 14 Phone (A & R)          |
|---------------|--------------------------------|----------|----------|---------------------|-----------------------------|-----------|
| LAM-20K       | VFHQ                          | TBD       | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K       | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)   | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K       | Our large dataset | TBD      | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |

**(A & R: Animating & Rendering)**

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

This project builds upon the amazing work of others:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

We are grateful for their contributions.

### Explore More

Discover our other projects:

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