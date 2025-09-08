# LAM: Large Avatar Model

**Create ultra-realistic, animatable 3D avatars from a single image in seconds with LAM (Large Avatar Model).**  Explore the future of digital human creation with LAM, a cutting-edge technology from Alibaba Group. [Find the original repository here](https://github.com/aigc3d/LAM).

*   **Key Features:**
    *   **One-Shot Avatar Generation:** Instantly create stunning 3D avatars from a single input image.
    *   **Fast Animation & Rendering:** Experience super-fast cross-platform animation and rendering, optimized for all devices.
    *   **Real-time Interaction:**  Low-latency SDK enables interactive chatting avatars for a seamless user experience.

## What is LAM?

LAM (Large Avatar Model) is a state-of-the-art model developed by Tongyi Lab, Alibaba Group, designed for the rapid creation and animation of 3D digital humans. This innovative technology allows users to transform a single image into a fully animatable avatar, revolutionizing how we interact with digital representations.

## Core Highlights

*   **Ultra-realistic Avatars:** Generate highly detailed and lifelike 3D avatars.
*   **Cross-platform Compatibility:** Animate and render your avatars on any device, from smartphones to high-end PCs.
*   **Real-time Chatting:** Integrate LAM avatars into real-time interactive chatting platforms with minimal delay.

## Latest Updates

*   **[May 20, 2025]:**  Released the [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)!
*   **[May 10, 2025]:**  The [ModelScope](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model) Demo now supports direct export of generated avatars for use in OpenAvatarChat!
*   **[April 30, 2025]:** Released an [Avatar Export Feature](tools/AVATAR_EXPORT_GUIDE.md) for seamless integration with OpenAvatarChat.
*   **[April 21, 2025]:** Launched the WebGL Interactive Chatting Avatar SDK on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat).
*   **[April 19, 2025]:** Released the [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model for audio-driven avatar animation.

## Get Started

### Online Demos

*   **Avatar Generation:**
    *   [Hugging Face Space](https://huggingface.co/spaces/3DAIGC/LAM)
    *   [ModelScope Space](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
*   **Interactive Chatting:**
    *   [Hugging Face Space](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
    *   [ModelScope Space](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

*   **Windows:** A one-click installation package is available for Windows (CUDA 12.8), supported by "十字鱼" - [Video](https://www.bilibili.com/video/BV13QGizqEey) &nbsp; &nbsp; [Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip).
*   **Linux:**
    ```bash
    git clone https://github.com/aigc3d/LAM.git
    cd LAM
    # Install with Cuda 12.1
    sh ./scripts/install/install_cu121.sh
    # Or Install with Cuda 11.8
    sh ./scripts/install/install_cu118.sh
    ```
*   **Windows Installation Guide:**  Refer to [Windows Install Guide](scripts/install/WINDOWS_INSTALL.md).

### Model Weights

| Model         | Training Data            | HuggingFace                                                                           | ModelScope                                                                                                                   | Reconstruction Time | A100 (A & R) | XiaoMi 14 Phone (A & R) |
|---------------|--------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|---------------------|-----------------------------|---------------------------|
| LAM-20K       | VFHQ                     | TBD                                                                                   | TBD                                                                                                                         | 1.4 s               | 562.9FPS                    | 110+FPS                   |
| LAM-20K       | VFHQ + NeRSemble         | [Link](https://huggingface.co/3DAIGC/LAM-20K)                                            | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)                                                                      | 1.4 s               | 562.9FPS                    | 110+FPS                   |
| LAM-20K       | Our large dataset        | TBD                                                                                   | TBD                                                                                                                         | 1.4 s               | 562.9FPS                    | 110+FPS                   |

(**A & R:** Animating & Rendering )

#### Download Instructions

**Hugging Face:**

```bash
# Download Assets
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

**ModelScope:**

```bash
pip3 install modelscope
# Download Assets
modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
# Download Model Weights
modelscope download "Damo_XR_Lab/LAM-20K" --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
```

### Run Gradio Demo

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

## Acknowledgements

This project builds upon the research and open-source contributions of:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

## Explore More

Discover our other projects:

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