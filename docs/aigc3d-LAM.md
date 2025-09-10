# LAM: Large Avatar Model - Create Animatable 3D Avatars from a Single Image

**LAM (Large Avatar Model) revolutionizes 3D avatar creation, allowing you to generate incredibly realistic, animatable avatars from a single image in seconds!** Check out the original repository for more details: [https://github.com/aigc3d/LAM](https://github.com/aigc3d/LAM)

[![Website](https://img.shields.io/badge/🏠-Website-blue)](https://aigc3d.github.io/projects/LAM/) 
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2502--17796-green)](https://arxiv.org/pdf/2502.17796)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
[![ModelScope](https://img.shields.io/badge/🧱-ModelScope-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model) 
[![Apache License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

## Key Features:

*   **One-Shot Avatar Generation:** Create ultra-realistic 3D avatars from a single input image in seconds.
*   **Cross-Platform Animation & Rendering:** Animate and render your avatars quickly on a wide range of devices.
*   **Real-time Interactive Avatars:** Utilize the low-latency SDK for engaging interactive chatting experiences.

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

## What's New

*   **May 20, 2025:** [WebGL-Render](https://github.com/aigc3d/LAM_WebRender) released for enhanced cross-platform compatibility.
*   **May 10, 2025:** ModelScope Demo enhanced to export avatars directly for OpenAvatarChat integration.
*   **April 30, 2025:** Avatar Export Feature [tools/AVATAR_EXPORT_GUIDE.md] released, enabling chatting with LAM avatars on OpenAvatarChat.
*   **April 21, 2025:** WebGL Interactive Chatting Avatar SDK released on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat).
*   **April 19, 2025:** [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model for animating avatars with audio input.

## To Do List:

*   [x] Release LAM-small trained on VFHQ and Nersemble.
*   [x] Release Huggingface space.
*   [x] Release Modelscope space.
*   [ ] Release LAM-large trained on a self-constructed large dataset.
*   [x] Release WebGL Render for cross-platform animation and rendering.
*   [x] Release audio driven model: Audio2Expression.
*   [x] Release Interactive Chatting Avatar SDK with [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat), including LLM, ASR, TTS, Avatar.

## Get Started

### Online Demos

*   **Avatar Generation:**
    *   [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
    *   [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
*   **Interactive Chatting:**
    *   [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
    *   [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

**One-Click Installation (Windows):** A user-friendly installation package is available for Windows with Cuda 12.8 support.

*   [Video](https://www.bilibili.com/video/BV13QGizqEey)
*   [Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)

**Linux Installation:**

```bash
git clone https://github.com/aigc3d/LAM.git
cd LAM
# Install with Cuda 12.1
sh ./scripts/install/install_cu121.sh
# Or Install with Cuda 11.8
sh ./scripts/install/install_cu118.sh
```

**Windows Installation:** Please refer to the [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

### Model Weights and Performance

| Model      | Training Data                  | HuggingFace | ModelScope | Reconstruction Time | A100 (A & R) | XiaoMi 14 Phone (A & R) |
|------------|--------------------------------|-------------|------------|---------------------|-----------------------------|-----------|
| LAM-20K    | VFHQ                          | TBD          | TBD        | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K    | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)   | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K    | Our large dataset | TBD         | TBD        | 1.4 s               | 562.9FPS                    | 110+FPS   |

**(A & R: Animating & Rendering)**

### Model Downloads

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
To export ZIP files for real-time conversations on OpenAvatarChat, see the [Guide](tools/AVATAR_EXPORT_GUIDE.md).
```bash
python app_lam.py --blender_path /path/blender
```

### Inference

```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

### Acknowledgements

LAM builds upon and acknowledges the contributions of these influential research works:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

### Explore More

Discover our other related projects:

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