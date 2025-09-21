<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

</div>

# MiniCPM-V & MiniCPM-o: Powerful Multimodal LLMs for On-Device Understanding

MiniCPM-V and MiniCPM-o are leading-edge multimodal Large Language Models (MLLMs) designed for efficient on-device deployment, offering exceptional performance in understanding images, videos, and audio, with text as output. **Experience cutting-edge multimodal AI capabilities on your device – find the models and documentation on [GitHub](https://github.com/OpenBMB/MiniCPM-V)!**

## Key Features

*   **MiniCPM-V 4.5: The Pinnacle of Vision-Language Capabilities**
    *   **Outperforms Leading Models:** Achieves state-of-the-art vision-language performance, **surpassing GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B** with only 8B parameters.
    *   **Efficient Video Understanding:** Supports high-FPS and long video analysis with up to a 96x compression rate for video tokens, including high FPS(up to 10FPS) video understanding and long video understanding capabilities.
    *   **Advanced Features:** Includes controllable hybrid fast/deep thinking, strong handwritten OCR, and complex table/document parsing.
    *   **User-Friendly:** Offers easy integration through llama.cpp, ollama, vLLM, and other frameworks, with int4, GGUF, AWQ quantizations, and quick local WebUI demos.

*   **MiniCPM-o 2.6: Revolutionizing On-Device Multimodal Experiences**
    *   **Exceptional Multimodal Performance:** Achieves performance comparable to GPT-4o on vision and speech, making it one of the most versatile open-source models.
    *   **Real-time Multimodal Streaming:** Supports multimodal live streaming on end-side devices, like iPads.
    *   **Multilingual and Trustworthy:** Incorporates trustworthy behaviors and multilingual support for a wide range of applications.
    *   **Bilingual Speech Conversation:** Supports bilingual real-time speech conversation, with configurable voices, voice cloning and more.

## Core Capabilities

*   **Multimodal Input:** Accepts images, videos, and audio, along with text inputs.
*   **High-Quality Text Outputs:** Delivers detailed and accurate text responses.
*   **End-Side Deployment:** Optimized for efficient deployment on mobile and edge devices.
*   **Multilingual Support:** Offers support for 30+ languages.

## Benefits

*   **On-Device Processing:** Enables real-time processing and enhanced privacy by keeping data local.
*   **State-of-the-Art Performance:** Delivers superior results compared to many larger models.
*   **Versatile Applications:** Suitable for a wide range of applications including image and video understanding, speech recognition, and multimodal live streaming.
*   **Easy to Use and Deploy:** Supports multiple frameworks and quantization methods for easy integration.

## Key Techniques and Innovation

*   **Unified 3D-Resampler:** Enables efficient processing of high-density videos.
*   **Unified Learning for OCR and Document Knowledge:** Improves OCR capabilities and knowledge extraction from documents.
*   **Hybrid Fast/Deep Thinking:** Provides a balance between efficiency and performance.
*   **End-to-end Omni-modal Architecture:** Maximizes the use of multimodal knowledge.

## Contents

*   [MiniCPM-V 4.5](#minicpm-v-45)
    *   [Inference Efficiency](#inference-efficiency)
*   [MiniCPM-o 2.6](#minicpm-o-26)
*   [MiniCPM-V & o Cookbook](#minicpm-v--o-cookbook)
*   [Chat with Our Demo on Gradio 🤗](#chat-with-our-demo-on-gradio-)
*   [Inference](#inference)
    *   [Model Zoo](#model-zoo)
    *   [Multi-turn Conversation](#multi-turn-conversation)
        *   [Chat with Multiple Images](#chat-with-multiple-images)
        *   [In-context Few-shot Learning](#in-context-few-shot-learning)
        *   [Chat with Video](#chat-with-video)
        *   [Speech and Audio Mode](#speech-and-audio-mode)
        *   [Multimodal Live Streaming](#multimodal-live-streaming)
    *   [Inference on Multiple GPUs](#inference-on-multiple-gpus)
    *   [Inference on Mac](#inference-on-mac)
    *   [Efficient Inference with llama.cpp, Ollama, vLLM](#efficient-inference-with-llamacpp-ollama-vllm)
*   [Fine-tuning](#fine-tuning)
*   [Awesome work using MiniCPM-V & MiniCPM-o](#awesome-work-using-minicpm-v--minicpm-o)
*   [FAQs](#faqs)
*   [Limitations](#limitations)
*   [Model License](#model-license)
*   [Institutions](#institutions)
*   [🌟 Star History](#-star-history)
*   [Key Techniques and Other Multimodal Projects](#key-techniques-and-other-multimodal-projects)
*   [Citation](#citation)

## Resources

*   [MiniCPM-V & o Cookbook](#minicpm-v--o-cookbook)
*   [Online Demo](https://minicpm-omni-webdemo-us.modelbest.cn/)
*   [Demo](http://120.92.209.146:8887/)
*   [Hugging Face Demo](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5)
*   [MiniCPM-V 2.0 Demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)