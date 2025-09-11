# LAM: Create Realistic 3D Avatars in Seconds 

**LAM (Large Avatar Model)** offers a revolutionary approach to create and animate lifelike 3D avatars from a single image, setting a new standard for speed and realism. Explore the power of LAM with our [original repo](https://github.com/aigc3d/LAM).

## Key Features:

*   **One-Shot Avatar Creation:** Generate ultra-realistic 3D avatars from a single image in a matter of seconds.
*   **Real-time Animation & Rendering:** Experience super-fast cross-platform animation and rendering on any device.
*   **Low-Latency SDK:** Utilize the low-latency SDK for real-time interactive chatting avatars.
*   **Cross-Platform Compatibility:** Supports both Windows and Linux with detailed setup guides.
*   **Integration with OpenAvatarChat:**  Easily export avatars for interactive chatting on the OpenAvatarChat platform.

## LAM in Action:

<div align="center">
  <video controls src="https://github.com/user-attachments/assets/98f66655-e1c1-40a9-ab58-bdd49dafedda" width="80%">
  </video>
</div>

## What's New:

*   **[May 20, 2025]:** WebGL-Render released ([WebGL-Render](https://github.com/aigc3d/LAM_WebRender))
*   **[May 10, 2025]:** ModelScope Demo now supports direct avatar export for OpenAvatarChat. ([ModelScope](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model))
*   **[April 30, 2025]:** Avatar Export Feature released ([Avatar Export Guide](tools/AVATAR_EXPORT_GUIDE.md)).
*   **[April 21, 2025]:** WebGL Interactive Chatting Avatar SDK released on OpenAvatarChat ([OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)).
*   **[April 19, 2025]:** Audio2Expression model released ([Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression)).

## Get Started:

### Online Demos:

*   **Hugging Face Space:** ([Hugging Face](https://huggingface.co/spaces/3DAIGC/LAM))
*   **ModelScope Space:** ([ModelScope](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model))
*   **OpenAvatarChat Demo:** ([Hugging Face](https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat)), ([ModelScope](https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat))

### Environment Setup:

*   **Windows:** One-click installation package ([Download Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/Installation/LAM-windows-one-click-install.zip)).
*   **Linux:** Follow the installation script in the provided instructions.

    ```bash
    git clone https://github.com/aigc3d/LAM.git
    cd LAM
    # Install with Cuda 12.1
    sh ./scripts/install/install_cu121.sh
    # Or Install with Cuda 11.8
    sh ./scripts/install/install_cu118.sh
    ```
*   **Windows Install Guide:**  Refer to the ([Windows Install Guide](scripts/install/WINDOWS_INSTALL.md)).

### Model Weights & Download:

**Model Performance:**

| Model   | Training Data                  | HuggingFace | ModelScope | Reconstruction Time | A100 (A & R) |   XiaoMi 14 Phone (A & R)          |
|---------|--------------------------------|----------|----------|---------------------|-----------------------------|-----------|
| LAM-20K | VFHQ                          | TBD       | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://www.modelscope.cn/models/Damo_XR_Lab/LAM-20K/summary)   | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | Our large dataset | TBD      | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |

*   **Hugging Face Download:**

    ```bash
    # Download Assets
    huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
    tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
    tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
    # Download Model Weights
    huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
    ```

*   **ModelScope Download:**

    ```bash
    pip3 install modelscope
    # Download Assets
    modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
    tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
    tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
    # Download Model Weights
    modelscope download "Damo_XR_Lab/LAM-20K" --local_dir "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
    ```

### Run the Application:

*   **Gradio Run:**
    ```bash
    python app_lam.py
    ```
    To export for OpenAvatarChat, refer to the ([Guide](tools/AVATAR_EXPORT_GUIDE.md)).
    ```bash
    python app_lam.py --blender_path /path/blender
    ```
*   **Inference:**
    ```bash
    sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
    ```

## Acknowledgements:

LAM builds upon the work of several open-source projects and research papers, including:

*   [OpenLRM](https://github.com/3DTopia/OpenLRM)
*   [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
*   [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
*   [VHAP](https://github.com/ShenhanQian/VHAP)

## Explore More:

*   **LHM (Looking at Human Model):** ([LHM](https://github.com/aigc3d/LHM))

## Citation:

```
@inproceedings{he2025lam,
  title={LAM: Large Avatar Model for One-shot Animatable Gaussian Head},
  author={He, Yisheng and Gu, Xiaodong and Ye, Xiaodan and Xu, Chao and Zhao, Zhengyi and Dong, Yuan and Yuan, Weihao and Dong, Zilong and Bo, Liefeng},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages={1--13},
  year={2025}
}