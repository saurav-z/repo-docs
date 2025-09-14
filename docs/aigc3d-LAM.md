# LAM: Create Interactive 3D Avatars from a Single Image (Official Pytorch Implementation)

**LAM (Large Avatar Model)** empowers you to generate and animate ultra-realistic 3D avatars from a single image in mere seconds! This repository provides the official PyTorch implementation.  See the original repo for more details: [https://github.com/aigc3d/LAM](https://github.com/aigc3d/LAM).

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

### Key Features:

*   **One-Shot Avatar Generation:** Instantly create a 3D avatar from a single input image.
*   **Fast Animation & Rendering:** Experience cross-platform animation and rendering on various devices.
*   **Real-time Interaction:** Utilize the low-latency SDK for interactive chatting avatars.
*   **High-Quality Results:** Achieve ultra-realistic 3D avatar appearances.
*   **Flexible Deployment:** Run on various hardware including phones.

<p align="center">
  <img src="./assets/images/teaser.jpg" width="100%">
</p>

## Core Highlights 🔥🔥🔥

*   **Ultra-realistic 3D Avatar Creation from One Image in Seconds**
*   **Super-fast Cross-platform Animating and Rendering on Any Devices**
*   **Low-latency SDK for Realtime Interactive Chatting Avatar**

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>


## 🚀 Get Started

### Online Demos

Experience the LAM technology:

*   **Hugging Face Space:**
    *   Avatar Generation:  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
    *   Interactive Chatting: [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
*   **ModelScope Space:**
    *   Avatar Generation: [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
    *   Interactive Chatting: [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

#### One-Click Installation (Windows)

A simplified, one-click installation package is available for Windows (Cuda 12.8) users, supported by "十字鱼".

*   [Video Tutorial](https://www.bilibili.com/video/BV13QGizqEey)
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

For Windows, refer to the [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

### Model Weights

| Model         | Training Data                  | HuggingFace                                  | ModelScope                                                                               | Reconstruction Time | A100 (A & R) | XiaoMi 14 Phone (A & R) |
|---------------|--------------------------------|-----------------------------------------------|------------------------------------------------------------------------------------------|---------------------|-----------------------------|--------------------------|
| LAM-20K       | VFHQ                          | TBD                                           | TBD                                                                                      | 1.4 s               | 562.9FPS                    | 110+FPS                  |
| LAM-20K       | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)                     | 1.4 s               | 562.9FPS                    | 110+FPS                  |
| LAM-20K       | Our large dataset             | TBD                                           | TBD                                                                                      | 1.4 s               | 562.9FPS                    | 110+FPS                  |

(**A & R:** Animating & Rendering )

#### Downloading Models

##### Hugging Face:

```bash
# Download Assets
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

##### ModelScope:

```bash
pip3 install modelscope
# Download Assets
modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
modelscope download "Damo_XR_Lab/LAM-20K" --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
```

### Running the Demo

#### Gradio Run
```bash
python app_lam.py
```
If you want to export ZIP files for real-time conversations on OpenAvatarChat, please refer to the [Guide](tools/AVATAR_EXPORT_GUIDE.md).
```bash
python app_lam.py --blender_path /path/blender
```

#### Inference
```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

### News & Updates

*   **[September 9, 2025]** Released the technical report of [PanoLAM](https://arxiv.org/pdf/2509.07552)!
*   **[May 20, 2025]** Released the [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)!
*   **[May 10, 2025]** ModelScope Demo supports exporting avatars for OpenAvatarChat!
*   **[April 30, 2025]** Released [Avatar Export Feature](tools/AVATAR_EXPORT_GUIDE.md)!
*   **[April 21, 2025]** Released WebGL Interactive Chatting Avatar SDK on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)!
*   **[April 19, 2025]** Released [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model!

### Acknowledgements

LAM is built upon the shoulders of these great research works and open-source projects:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

### More Works

Explore our other related projects:

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