# EchoMimicV3: Animate Humans with Just 1.3 Billion Parameters 

EchoMimicV3 revolutionizes human animation by offering unified multi-modal and multi-task capabilities with a surprisingly compact model. Explore the technology on the original repository: [https://github.com/antgroup/echomimic_v3](https://github.com/antgroup/echomimic_v3)

**Key Features:**

*   **Unified Approach:** Handles multiple modalities and tasks for comprehensive human animation.
*   **Compact Size:** Achieves impressive results with only 1.3 billion parameters, making it efficient.
*   **Gradio Demo:**  Interactive demo available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3)
*   **Low VRAM Requirement:** Generate videos with as little as 12GB of VRAM and ComfyUI support for 16GB.
*   **Easy to Use:** Includes Quick Start instructions and pre-trained models for immediate use.
*   **Multiple Resources:** Access models on Hugging Face and ModelScope, plus a project page and paper link.

**What's New:**

*   **Model Release:**  Models are available on Hugging Face and ModelScope.
*   **Demo:**  Gradio Demo available on ModelScope
*   **Video Generation:**  Generate videos with 12 GB of VRAM using the included GradioUI or 16GB with ComfyUI.
*   **Paper:** Access the [paper](https://arxiv.org/abs/2507.03905) on arXiv.
*   **Code Release:** Codes are released on GitHub

**Get Started Quickly:**

1.  **Environment Setup:**
    *   Tested on: Centos 7.2/Ubuntu 22.04, Cuda >= 12.1.
    *   Supported GPUs: A100(80G) / RTX4090D (24G) / V100(16G).
    *   Python Versions: 3.10 / 3.11.
2.  **Installation (Windows):** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).
3.  **Installation (Linux):**
    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    pip install -r requirements.txt
    ```
4.  **Model Preparation:** Download and place models as instructed (links provided).
5.  **Quick Inference:**  Run `python infer.py` or for GradioUI, run `python app_mm.py`.

**Tips for Optimal Results:**

*   **Audio CFG:** Use `audio_guidance_scale` between 2-3 for lip sync and quality balance.
*   **Text CFG:** Use `guidance_scale` between 3-6 for prompt following and quality balance.
*   **TeaCache:** Use a threshold of 0-0.1.
*   **Sampling Steps:** 5 steps for talking heads, 15-25 for full-body animation.
*   **Long Video Generation:** Utilize Long Video CFG.
*   **VRAM Optimization:** Adjust `partial_video_length` (e.g., 81, 65) for lower VRAM use.

**Example Outputs:**
Gallery of Sample Video outputs, with image tags from the original readme.

**Coming Soon:**

*   720P Pretrained models
*   The training code of EchoMimicV3

**EchoMimic Series:**

*   [EchoMimicV3](https://github.com/antgroup/echomimic_v3)
*   [EchoMimicV2](https://github.com/antgroup/echomimic_v2)
*   [EchoMimicV1](https://github.com/antgroup/echomimic)

**Citation:**

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

**License:** Apache 2.0 License. Please adhere to the license when using the model.

**Star History:**
Graph of Star History is included.
```
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)