# LAM: Revolutionizing 3D Avatar Creation with One-Shot Animatable Gaussian Heads

**Create stunning, interactive 3D avatars from a single image in seconds with LAM, the Large Avatar Model!** Explore the cutting-edge research and capabilities of LAM, the official PyTorch implementation, and bring your digital presence to life.  [Visit the original repository](https://github.com/aigc3d/LAM).

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

*   **[Paper](https://arxiv.org/pdf/2502.17796)**
*   **[Website](https://aigc3d.github.io/projects/LAM/)**
*   **[Hugging Face Demo](https://huggingface.co/spaces/3DAIGC/LAM)**
*   **[ModelScope Demo](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)**
*   **[License](https://www.apache.org/licenses/LICENSE-2.0)**

<p align="center">
  <img src="./assets/images/teaser.jpg" width="100%">
</p>

## Key Features

*   **One-Shot Avatar Generation:** Generate ultra-realistic 3D avatars from a single image in mere seconds.
*   **Cross-Platform Animation & Rendering:** Experience super-fast animation and rendering on any device.
*   **Real-Time Interactive Chatting Avatar:** Utilize our low-latency SDK for interactive experiences.
*   **OpenAvatarChat Integration:**  Easily export avatars for use with the OpenAvatarChat platform.
*   **Audio-Driven Animation:** Animate your avatars with audio input using the Audio2Expression model.

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

## What's New

*   **[May 20, 2025]** Released [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)
*   **[May 10, 2025]** ModelScope Demo now supports direct avatar export to OpenAvatarChat
*   **[April 30, 2025]** Released [Avatar Export Feature](tools/AVATAR_EXPORT_GUIDE.md) for OpenAvatarChat compatibility.
*   **[April 21, 2025]** Released WebGL Interactive Chatting Avatar SDK on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat).
*   **[April 19, 2025]** Released [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) for audio-driven animation.

## Get Started

### Online Demo

Experience the power of LAM with our online demos:

*   **Avatar Generation:**
    *   [Hugging Face Space](https://huggingface.co/spaces/3DAIGC/LAM)
    *   [ModelScope Space](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
*   **Interactive Chatting:**
    *   [Hugging Face Space](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
    *   [ModelScope Space](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

#### One-Click Installation (Windows)

We provide a one-click installation package on Windows (Cuda 12.8), supported by "十字鱼". &nbsp; &nbsp;
[Video](https://www.bilibili.com/video/BV13QGizqEey) &nbsp; &nbsp;
[Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)

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

| Model   | Training Data                  | HuggingFace                                      | ModelScope                                                   | Reconstruction Time | A100 (A & R) |  XiaoMi 14 Phone (A & R)           |
|---------|--------------------------------|--------------------------------------------------|--------------------------------------------------------------|---------------------|-----------------------------|-----------|
| LAM-20K | VFHQ                          | TBD                                              | TBD                                                          | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary) | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | Our large dataset | TBD      | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |

(**A & R:** Animating & Rendering )

#### Download Model Weights

```bash
# HuggingFace Download
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/

# ModelScope Download
pip3 install modelscope
modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
modelscope download "Damo_XR_Lab/LAM-20K" --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
```

### Run the Demo

```bash
python app_lam.py
```

To export ZIP files for OpenAvatarChat, refer to the [Guide](tools/AVATAR_EXPORT_GUIDE.md).

```bash
python app_lam.py --blender_path /path/blender
```

### Inference

```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

## Acknowledgements

This project leverages the following excellent works:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

## More Works

Explore our other projects:

*   [LHM](https://github.com/aigc3d/LHM)

## Citation

```
@inproceedings{he2025lam,
  title={LAM: Large Avatar Model for One-shot Animatable Gaussian Head},
  author={He, Yisheng and Gu, Xiaodong and Ye, Xiaodan and Xu, Chao and Zhao, Zhengyi and Dong, Yuan and Yuan, Weihao and Dong, Zilong and Bo, Liefeng},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages={1--13},
  year={2025}
}
```