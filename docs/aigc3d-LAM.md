# LAM: Create Stunning 3D Avatars from a Single Image with Pytorch (Official Implementation)

**Effortlessly transform a single image into a dynamic, animatable 3D avatar in seconds** with LAM (Large Avatar Model), a cutting-edge technology for realistic digital human creation.  This repository contains the official PyTorch implementation of LAM.  For more details, visit the original repository: [https://github.com/aigc3d/LAM](https://github.com/aigc3d/LAM)

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

### Core Features

*   **One-Shot Avatar Generation:** Create ultra-realistic 3D avatars from a single image in mere seconds.
*   **Cross-Platform Compatibility:** Enjoy super-fast animation and rendering on any device.
*   **Real-time Interaction:** Leverage a low-latency SDK for interactive chatting avatars.

### Key Highlights

*   **Speed and Efficiency:** Fast avatar creation and real-time rendering.
*   **Realism:** High-quality 3D avatars with natural-looking details.
*   **Interactive Capabilities:** Integrate your avatars into real-time chat and communication platforms.

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

### Key Updates & News

*   **September 9, 2025:** Released the technical report of [PanoLAM](https://arxiv.org/pdf/2509.07552)!
*   **May 20, 2025:** Released the [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)!
*   **May 10, 2025:** The [ModelScope](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model) Demo now supports direct export of avatars for OpenAvatarChat.
*   **April 30, 2025:**  Released an [Avatar Export Feature](tools/AVATAR_EXPORT_GUIDE.md) for use with OpenAvatarChat.
*   **April 21, 2025:** Launched the WebGL Interactive Chatting Avatar SDK on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat).
*   **April 19, 2025:** Released the [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model for audio-driven avatar animation.

### Get Started

#### Online Demos

*   **Hugging Face Space:** Generate avatars from a single image: [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
*   **ModelScope Space:** Generate avatars from a single image:  [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
*   **OpenAvatarChat Integration:** Interact with your avatars in real time:
    *   [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
    *   [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

#### Environment Setup

*   **Windows (One-Click Installation):**  [Video](https://www.bilibili.com/video/BV13QGizqEey) &nbsp; &nbsp; [Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)
*   **Linux:**

```bash
git clone https://github.com/aigc3d/LAM.git
cd LAM
# Install with Cuda 12.1
sh ./scripts/install/install_cu121.sh
# Or Install with Cuda 11.8
sh ./scripts/install/install_cu118.sh
```

*   **Windows:**  Refer to the [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

#### Model Weights

| Model       | Training Data                  | HuggingFace                                                                | ModelScope                                                                 | Reconstruction Time | A100 (A & R)  | XiaoMi 14 Phone (A & R) |
|-------------|--------------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------|---------------|---------------------------|
| LAM-20K     | VFHQ                           | TBD                                                                        | TBD                                                                         | 1.4 s               | 562.9FPS      | 110+FPS                   |
| LAM-20K     | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K)                                | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)        | 1.4 s               | 562.9FPS      | 110+FPS                   |
| LAM-20K     | Our large dataset | TBD                                                                        | TBD                                                                         | 1.4 s               | 562.9FPS      | 110+FPS                   |

(**A & R:** Animating & Rendering )

#### Download Model Weights

*   **Hugging Face:**

```bash
# Download Assets
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

*   **ModelScope:**

```bash
pip3 install modelscope
# Download Assets
modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
modelscope download "Damo_XR_Lab/LAM-20K" --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
```

#### Run the Demo

*   **Gradio:**

```bash
python app_lam.py
```

*   **OpenAvatarChat Export:**

```bash
python app_lam.py --blender_path /path/blender
```

#### Inference

```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

### Acknowledgements

LAM builds upon the shoulders of giants.  We are grateful for the contributions of:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

### Explore More

Discover our other innovative projects:

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