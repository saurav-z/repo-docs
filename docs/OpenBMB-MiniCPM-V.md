<div align="center">
  <img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" alt="MiniCPM-V & MiniCPM-o Title">
</div>

# MiniCPM-V & MiniCPM-o: Powerful Multimodal LLMs for On-Device Understanding

**MiniCPM-V & MiniCPM-o** are cutting-edge, efficient, end-side multimodal LLMs (MLLMs) designed for robust image, video, audio, and text understanding, all on your phone.  **Explore the original repo for more details: [https://github.com/OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)**

**Key Features:**

*   **🔥 High Performance on Device**: Achieve GPT-4o-level performance on your mobile devices, unlocking advanced vision, audio, and video understanding capabilities.
*   **🖼️ Image & Video Understanding:** Process images, videos, and text inputs to generate high-quality text outputs. MiniCPM-V 4.5 brings efficient high-FPS and long video understanding.
*   **🗣️ Audio Integration**: MiniCPM-o additionally accepts audio as input and provides high-quality speech outputs, featuring bilingual real-time speech conversations, voice cloning, and role-play capabilities.
*   **⚡ Efficient Deployment**: Designed for efficient end-side deployment, allowing for powerful multimodal experiences directly on your devices.
*   **🌍 Multilingual Support**: Broad language support including but not limited to English and Chinese.
*   **🚀 Continuous Updates**:  Regularly updated models and feature releases to maximize performance.

## Highlights:

*   **[MiniCPM-V 4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5)**: Outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language capabilities. Introduces efficient high-FPS and long video understanding.
*   **[MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)**: Achieves comparable performance to GPT-4o-202405 in vision, speech, and multimodal live streaming. Supports bilingual real-time speech conversation, end-to-end voice cloning, and multimodal live streaming on end-side devices.

## Key Advantages:

*   **Strong Performance:** Leading performance in vision-language tasks, surpassing many proprietary and open-source models.
*   **On-Device Capability:** Enable multimodal understanding and interaction directly on your phone, offering real-time response and reduced latency.
*   **Versatile Applications:** Explore a wide range of applications, including image understanding, video analysis, audio processing, speech recognition, and multimodal live streaming.
*   **Easy to Use:**  Quickly get started with the integrated demos, tutorials and the Cookbook, offering a smooth onboarding experience.

## Get Started:

*   **[MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)**: Comprehensive guides and ready-to-deploy solutions for various use cases and hardware environments.
*   **[Online Demo](https://minicpm-omni-webdemo-us.modelbest.cn/)**: Experience the power of MiniCPM-o 2.6 & others in the cloud.
*   **Inference**

    *   **Model Zoo:** Available models (MiniCPM-V 4.5, MiniCPM-o 2.6 and more) for GPU/CPU, quantized models, and GGUF.
    *   **Multi-turn Conversation**
        *   Chat with Multiple Images
        *   In-context Few-shot Learning
        *   Chat with Video
        *   Speech and Audio Mode
        *   Multimodal Live Streaming
    *   **Efficient Inference with llama.cpp, Ollama, vLLM**
*   **Fine-tuning**: Unleash the potential by customizing your model!
    *   Simple Fine-tuning
    *   With Align-Anything
    *   With LLaMA-Factory
    *   With the SWIFT Framework
*   **[FAQs](./docs/faqs.md)**: Find answers to common questions.
*   **[Model License](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md)**:  Understand the usage terms and conditions.

## Supporting Resources:

*   [Technical Report (Coming Soon)]
*   **Social Media:**

    *   [WeChat](docs/wechat.md)
    *   [Discord](https://discord.gg/rftuRMbqzf)

## Citation:

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```