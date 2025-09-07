# LAM: Revolutionizing 3D Avatar Creation with One Image

Create incredibly realistic and animatable 3D avatars from a single image in seconds with **LAM: Large Avatar Model**, a cutting-edge PyTorch implementation developed by Tongyi Lab, Alibaba Group.  Explore the possibilities at the [original LAM repository](https://github.com/aigc3d/LAM).

[![Website](https://img.shields.io/badge/🏠-Website-blue)](https://aigc3d.github.io/projects/LAM/)
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2502--17796-green)](https://arxiv.org/pdf/2502.17796)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
[![ModelScope](https://img.shields.io/badge/🧱-ModelScope-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
[![Apache License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

## Key Features

*   **One-Shot Avatar Generation:** Instantly create stunning 3D avatars from a single input image.
*   **Real-Time Animation & Rendering:** Experience super-fast cross-platform animation and rendering on any device.
*   **Interactive Chatting Avatar SDK:** Utilize the low-latency SDK for seamless real-time interactive chatting avatar experiences.
*   **High-Performance:** Achieve impressive FPS on a variety of hardware, even mobile devices like the XiaoMi 14 phone.
*   **Open Source:** Build on amazing open-source projects

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

## What's New

*   **[May 20, 2025]** Released [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)!
*   **[May 10, 2025]** ModelScope demo supports direct export for OpenAvatarChat.
*   **[April 30, 2025]** Released the [Avatar Export Feature](tools/AVATAR_EXPORT_GUIDE.md).
*   **[April 21, 2025]** Launched the WebGL Interactive Chatting Avatar SDK on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)
*   **[April 19, 2025]** Released the [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model.

## Get Started

### Online Demos

*   **Avatar Generation:**
    *   [Hugging Face Space](https://huggingface.co/spaces/3DAIGC/LAM)
    *   [ModelScope Space](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)

*   **Interactive Chatting:**
    *   [Hugging Face Space](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
    *   [ModelScope Space](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Installation

#### One-Click Installation (Windows)

*   Download the one-click installation package (CUDA 12.8 support) from [here](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip).
*   See [Video](https://www.bilibili.com/video/BV13QGizqEey) for more information

#### Linux

```bash
git clone https://github.com/aigc3d/LAM.git
cd LAM
# Install with Cuda 12.1
sh ./scripts/install/install_cu121.sh
# Or Install with Cuda 11.8
sh ./scripts/install/install_cu118.sh
```

#### Windows

*   Refer to the [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

### Model Weights

| Model       | Training Data            | HuggingFace                                                                     | ModelScope                                                                   | Reconstruction Time | A100 (A & R) | XiaoMi 14 Phone (A & R) |
|-------------|--------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------|---------------------|-----------------------------|---------------------------|
| LAM-20K     | VFHQ                     | TBD                                                                             | TBD                                                                         | 1.4 s               | 562.9 FPS                    | 110+ FPS                  |
| LAM-20K     | VFHQ + NeRSemble         | [Link](https://huggingface.co/3DAIGC/LAM-20K)                                    | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)         | 1.4 s               | 562.9 FPS                    | 110+ FPS                  |
| LAM-20K     | Our large dataset        | TBD                                                                             | TBD                                                                         | 1.4 s               | 562.9 FPS                    | 110+ FPS                  |

(**A & R:** Animating & Rendering)

#### Download Model Weights
*Use the following instructions to download model weights:*

#### Hugging Face
```bash
# Download Assets
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

#### ModelScope
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

To export ZIP files for real-time conversations on OpenAvatarChat, refer to the [Guide](tools/AVATAR_EXPORT_GUIDE.md).

```bash
python app_lam.py --blender_path /path/blender
```

### Inference

```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

### Acknowledgements

This project builds upon the excellent work of the following:
-   [OpenLRM](https://github.com/3DTopia/OpenLRM)
-   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
-   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
-   [VHAP](https://github.com/ShenhanQian/VHAP)

### Explore More

Check out our other related works:
-   [LHM](https://github.com/aigc3d/LHM)

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