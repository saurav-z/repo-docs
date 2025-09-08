<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

</div>

# MiniCPM-V: The Revolutionary Multimodal LLM for On-Device AI

**Experience cutting-edge vision and audio understanding on your phone with MiniCPM-V, a series of efficient multimodal LLMs (MLLMs) delivering GPT-4o-level performance!**  [Explore the original repository](https://github.com/OpenBMB/MiniCPM-V) for full details.

## Key Features

*   **Unrivaled Performance:** MiniCPM-V 4.5 surpasses GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language capabilities. MiniCPM-o 2.6 matches GPT-4o-202405 on vision, speech, and multimodal live streaming.
*   **On-Device Efficiency:** Designed for efficient deployment on mobile devices, enabling real-time video and audio understanding on your phone.
*   **Multimodal Input:** Accepts images, videos, and text (MiniCPM-V) and additionally audio (MiniCPM-o) as input.
*   **High-Quality Output:** Generates high-quality text (MiniCPM-V) and speech outputs (MiniCPM-o).
*   **New Features:**
    *   **MiniCPM-V 4.5:** High-FPS and long video understanding, hybrid fast/deep thinking, strong OCR and document parsing.
    *   **MiniCPM-o 2.6:** Bilingual real-time speech conversation, emotion/style control, end-to-end voice cloning, and multimodal live streaming.
*   **Versatile Deployment:** Supported by llama.cpp, vLLM, Ollama, and SGLang. Quantized models (int4, GGUF, AWQ) are also available.
*   **Open Source:** Models, code, and datasets are open-sourced for academic research and free commercial use (after registration).

## Latest Updates

*   **[2024.08.26]** 🔥🔥🔥 **MiniCPM-V 4.5 Released!** The latest and most capable model in the series, offering enhanced vision-language capabilities.
*   **[2024.08.01]** ⭐️⭐️⭐️ [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook) released.
*   **[2024.01.13]** 🔥🔥🔥 **MiniCPM-o 2.6 Released!**  Achieving GPT-4o level performance with enhanced speech and live streaming capabilities.

## Key Techniques

*   **Unified 3D-Resampler (MiniCPM-V 4.5):** Enables efficient high-FPS and long video understanding.
*   **Unified Learning for OCR and Knowledge:** Improved OCR and document parsing.
*   **Hybrid Fast/Deep Thinking (MiniCPM-V 4.5):** For different user scenarios, balance efficiency and performance.
*   **End-to-end Omni-modal Architecture (MiniCPM-o 2.6):** Fully exploits multimodal knowledge and new mechanisms for multimodal live streaming.
*   **Configurable Speech Modeling Design (MiniCPM-o 2.6):** For flexible voice configurations.

## Quick Start

*   **Model Zoo:** [Access the latest models](https://github.com/OpenBMB/MiniCPM-V#model-zoo).
*   **Demos:** Try out the online demo of [MiniCPM-o 2.6](https://minicpm-omni-webdemo-us.modelbest.cn/) | [MiniCPM-V 2.6](http://120.92.209.146:8887/) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2).
*   **Documentation & Cookbook:** Comprehensive guides for diverse user scenarios with our new [Docs Site](https://minicpm-o.readthedocs.io/en/latest/index.html) and [Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook).

## Inference & Fine-tuning

*   **Inference:** Detailed instructions and code examples [here](https://github.com/OpenBMB/MiniCPM-V#inference).
*   **Fine-tuning:** Support for simple fine-tuning, Align-Anything, LLaMA-Factory, and SWIFT framework. Details [here](https://github.com/OpenBMB/MiniCPM-V#fine-tuning).

## Awesome Work
[Find community projects](https://github.com/OpenBMB/MiniCPM-V#awesome-work-using-minicpm-v--minicpm-o) utilizing MiniCPM-V.

## Limitations
[Read about limitations](https://github.com/OpenBMB/MiniCPM-V#limitations).

## Citation

If you find our model/code/paper helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```

## License & Statement

*   The project is licensed under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License.
*   The use of MiniCPM-o/V model weights must strictly follow [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).
*   The models and weights of MiniCPM are completely free for academic research. after filling out a ["questionnaire"](https://modelbest.feishu.cn/share/base/form/shrcnpV5ZT9EJ6xYjh3Kx0J6v8g) for registration, are also available for free commercial use.
*   Please see the [Statement](https://github.com/OpenBMB/MiniCPM-V#statement).

## Stay Connected

*   This project is developed by [THUNLP](https://nlp.csai.tsinghua.edu.cn/) and [ModelBest](https://modelbest.cn/).
*   🌟 Star History is available [here](https://github.com/OpenBMB/MiniCPM-V#-%EF%B8%8F-star-history).