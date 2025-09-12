<div align="center">
  <img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" alt="MiniCPM-V and MiniCPM-o Logo">
</div>

# MiniCPM-V & MiniCPM-o: Powerful Multimodal LLMs for On-Device Understanding

**Experience cutting-edge multimodal understanding on your phone with MiniCPM-V and MiniCPM-o!**  Explore the original repository [here](https://github.com/OpenBMB/MiniCPM-V).

## Key Features:

*   **On-Device Performance:** Designed for efficient deployment on mobile devices.
*   **Multimodal Input:** Accepts images, videos, text, and audio.
*   **High-Quality Output:** Delivers high-quality text and speech outputs.
*   **Open Source:** Explore and use freely.
*   **GPT-4o Level Performance:** Achieving state-of-the-art performance.
*   **Multilingual Support:** Supports over 30 languages.

## Core Models:

### [MiniCPM-V 4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5): The Leading Vision-Language Model

*   🔥 **Outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language capabilities.**
*   🎬 **Efficient High-FPS and Long Video Understanding:** New 3D-Resampler with 96x compression for video tokens.
*   ⚙️ **Controllable Hybrid Fast/Deep Thinking:** Offers both efficient and comprehensive reasoning modes.
*   💪 **Strong OCR and Document Parsing:** Based on LLaVA-UHD architecture for high-resolution image processing.
*   💫 **Easy Usage:** Support for llama.cpp, Ollama, vLLM, quantization, and fine-tuning.
    *   Try MiniCPM-V 4.5 Online Demo: [Server](http://101.126.42.235:30910/)

### [MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6): The All-in-One Vision-Speech Model

*   ⭐️ **Achieves comparable performance to GPT-4o-202405 in vision, speech, and multimodal live streaming.**
*   🎙 **Real-time Speech Conversation:** Bilingual support and configurable voices.
*   🎬 **Multimodal Live Streaming:** Supports end-to-end live streaming on mobile devices.
*   💪 **Strong OCR Capabilities** Enhanced capabilities in image processing.
*   🚀 **Superior Efficiency:** Improved token density for faster inference.
    *   Try MiniCPM-o 2.6 Online Demo: [Server](https://minicpm-omni-webdemo-us.modelbest.cn/)

## Key Techniques:

*   **Architechture: Unified 3D-Resampler for High-density Video Compression.**
*   **Pre-training: Unified Learning for OCR and Knowledge from Documents.**
*   **Post-training: Hybrid Fast/Deep Thinking with Multimodal RL.**

## Quick Start:

*   **Chat with Our Demo on Gradio 🤗:**  [MiniCPM-o 2.6](https://minicpm-omni-webdemo-us.modelbest.cn/) | [MiniCPM-V 2.6](http://120.92.209.146:8887/) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2).
*   **Inference:** Explore our [Model Zoo](#model-zoo) and comprehensive [Inference](#inference) guide, including multi-turn conversation, multi-image input, few-shot learning, and video processing.

## Resources:

*   [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook): Get up and running with our detailed guides.
*   [FAQs](./docs/faqs.md): Find answers to frequently asked questions.
*   [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md): Details on model usage.

## Support and Community:

*   [WeChat](docs/wechat.md)
*   [Discord](https://discord.gg/rftuRMbqzf)

## Citation

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```

##  Institutions

This project is developed by the following institutions:

- <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)

## Disclaimer
See [Statement](./#statement)