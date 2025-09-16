# LAM: Create Realistic 3D Avatars from a Single Image in Seconds!

LAM (Large Avatar Model) is a cutting-edge PyTorch implementation enabling the creation of highly realistic and animatable 3D avatars from a single image. Developed by Tongyi Lab, Alibaba Group, LAM offers unparalleled speed and cross-platform compatibility for next-generation interactive experiences. Explore the project on [GitHub](https://github.com/aigc3d/LAM).

## Key Features

*   **One-Shot Avatar Generation:** Generate stunning 3D avatars from a single image in mere seconds.
*   **Real-time Animation & Rendering:** Experience super-fast cross-platform animation and rendering capabilities, optimized for any device.
*   **Low-Latency SDK:** Integrate seamlessly with real-time interactive chatting applications.
*   **Interactive Chatting:** Integrate with OpenAvatarChat to chat with your LAM-generated 3D digital humans.
*   **Audio-Driven Animation:** Animate your avatar with audio input using the included Audio2Expression model.

## Core Highlights 🔥🔥🔥

*   **Ultra-realistic 3D Avatar Creation from One Image in Seconds**
*   **Super-fast Cross-platform Animating and Rendering on Any Devices**
*   **Low-latency SDK for Realtime Interactive Chatting Avatar**

## News

*   **[September 9, 2025]** Released the technical report of [PanoLAM](https://arxiv.org/pdf/2509.07552)!
*   **[May 20, 2025]** Released the [WebGL-Render](https://github.com/aigc3d/LAM_WebRender)!
*   **[May 10, 2025]** The [ModelScope](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model) Demo now supports direct export to OpenAvatarChat.
*   **[April 30, 2025]** Released [Avatar Export Feature](tools/AVATAR_EXPORT_GUIDE.md).
*   **[April 21, 2025]** Released WebGL Interactive Chatting Avatar SDK on [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)
*   **[April 19, 2025]** Released the [Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) model.

## To Do List

*   \[x] Release LAM-small trained on VFHQ and Nersemble.
*   \[x] Release Huggingface space.
*   \[x] Release Modelscope space.
*   \[ ] Release LAM-large trained on a self-constructed large dataset.
*   \[x] Release WebGL Render for cross-platform animation and rendering.
*   \[x] Release audio driven model: Audio2Expression.
*   \[x] Release Interactive Chatting Avatar SDK with [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat), including LLM, ASR, TTS, Avatar.

## Get Started

### Online Demo

*   **Hugging Face Space:** [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
*   **ModelScope Space:** [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model)
*   **Interactive Chatting Demo:**
    *   **Hugging Face Space:** [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)
    *   **ModelScope Space:** [![ModelScope](https://img.shields.io/badge/🧱-ModelScope_Space-blue)](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat)

### Environment Setup

*   **One-Click Installation (Windows):**  A simplified one-click installer for Windows (Cuda 12.8) is available.
    *   [Video](https://www.bilibili.com/video/BV13QGizqEey)
    *   [Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)
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

### Model Weights

| Model     | Training Data          | HuggingFace   | ModelScope                             | Reconstruction Time | A100 (A & R) | XiaoMi 14 Phone (A & R) |
| --------- | ----------------------- | ------------- | -------------------------------------- | --------------------- | -------------------------- | ----------------------- |
| LAM-20K   | VFHQ                    | TBD           | TBD                                    | 1.4 s                 | 562.9 FPS                  | 110+ FPS                |
| LAM-20K   | VFHQ + NeRSemble          | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary) | 1.4 s                 | 562.9 FPS                  | 110+ FPS                |
| LAM-20K   | Our large dataset       | TBD           | TBD                                    | 1.4 s                 | 562.9 FPS                  | 110+ FPS                |

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

*   For exporting ZIP files for real-time conversations on OpenAvatarChat, follow the [Guide](tools/AVATAR_EXPORT_GUIDING.md).
```bash
python app_lam.py --blender_path /path/blender
```

### Inference

```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

## Acknowledgements

This project leverages the contributions of several open-source projects:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

We appreciate their excellent work and contributions.

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