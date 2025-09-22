# LAM: Generate Realistic 3D Avatars from a Single Image

**Create stunning, interactive 3D avatars from a single image in seconds with the Large Avatar Model (LAM).**  [Explore the original repository](https://github.com/aigc3d/LAM) for the official PyTorch implementation.

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

## Key Features

*   **One-Shot Avatar Generation:** Instantly create ultra-realistic 3D avatars from a single input image.
*   **Real-time Animation & Rendering:** Experience super-fast cross-platform animation and rendering, optimized for any device.
*   **Low-Latency SDK:** Integrate the LAM model into your applications with a low-latency SDK for real-time interactive chatting avatars.
*   **Cross-Platform Compatibility:** The model is compatible with various platforms via the WebGL-Render.
*   **Audio-Driven Animation:** Animate your avatar with audio input using the integrated Audio2Expression model.

## What's New

*   **[September 9, 2025]:** Released the technical report of [PanoLAM](https://arxiv.org/pdf/2509.07552)!
*   **[May 20, 2025]:** Released the [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)!
*   **[May 10, 2025]:** The [ModelScope](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model) Demo now supports directly exporting the generated Avatar to files required by OpenAvatarChat for interactive chatting!
*   **[April 30, 2025]:** Released an [Avatar Export Feature](tools/AVATAR_EXPORT_GUIDE.md) for use with OpenAvatarChat.
*   **[April 21, 2025]:** Released the WebGL Interactive Chatting Avatar SDK on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat).
*   **[April 19, 2025]:** Released the [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model.

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

## Get Started

### Online Demos

*   **Hugging Face Space:** Generate avatars: [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
*   **ModelScope Space:** Generate avatars: [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
*   **OpenAvatarChat:** Interactive Chatting : [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
*   **OpenAvatarChat:** Interactive Chatting : [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

**One-Click Installation (Windows):**  Simplified setup for Windows with CUDA 12.8 support via "十字鱼".  [Video](https://www.bilibili.com/video/BV13QGizqEey) &nbsp; &nbsp;  [Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)

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

Refer to the detailed [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

### Model Weights

| Model   | Training Data                  | HuggingFace | ModelScope | Reconstruction Time | A100 (A & R) | XiaoMi 14 Phone (A & R) |
|---------|--------------------------------|----------|----------|---------------------|-----------------------------|-----------|
| LAM-20K | VFHQ                          | TBD       | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)   | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | Our large dataset | TBD      | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |

**(A & R: Animating & Rendering)**

#### Download from Hugging Face

```bash
# Download Assets
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

#### Download from ModelScope

```bash
pip3 install modelscope
# Download Assets
modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
modelscope download "Damo_XR_Lab/LAM-20K" --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
```

### Run the Gradio Demo

```bash
python app_lam.py
```

To export ZIP files for OpenAvatarChat, see the [Guide](tools/AVATAR_EXPORT_GUIDE.md).

```bash
python app_lam.py --blender_path /path/blender
```

### Inference

```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

### Acknowledgements

This project builds upon the following research and open-source projects:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

Thank you to the contributors for their valuable contributions.

### Explore More

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