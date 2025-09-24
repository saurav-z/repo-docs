<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

</div>

# MiniCPM-V: Powerful Multimodal LLMs for On-Device AI

**MiniCPM-V** is a series of cutting-edge, efficient multimodal large language models (MLLMs) designed for **understanding images, videos, and text with remarkable performance, even on your phone**.  This repository contains the resources for all versions of the MiniCPM-V series and MiniCPM-o models.  Explore the latest advancements and capabilities of this innovative project.

**[➡️ Go to Original Repo](https://github.com/OpenBMB/MiniCPM-V)**

## Key Features

*   🎯 **Unrivaled Performance:** Experience vision-language capabilities that rival GPT-4o, Gemini-2.0 Pro, and Qwen2.5-VL 72B, all within an efficient and on-device model.
*   🎬 **Advanced Video Understanding:** Achieve high-FPS (up to 10FPS) and long video understanding with a 96x compression rate for video tokens.
*   🗣️ **Multimodal Live Streaming:**  Interact with real-time video and audio streams, for various live streaming scenarios.
*   📝 **Enhanced OCR & Document Processing:**  Process high-resolution images, including strong handwritten OCR and complex table/document parsing.
*   🌐 **Multilingual Support:** Interact with the model in over 30 languages.
*   📱 **On-Device Deployment:** Optimized for efficient inference on your phone, iPad, and other end-side devices.
*   🎤 **End-to-End Speech and Audio Interaction**: Leverage bilingual, real-time speech conversation with voice cloning.
*   🤝 **Trustworthy Behaviors**: Outperform GPT-4o-latest on MMHal-Bench.
*   👩‍💻 **Extensive Community Support:** Access to a cookbook with deployment solutions, online and local demos powered by Gradio,  and a dedicated discord channel.

## Models

### MiniCPM-V 4.5 (Latest)

*   **Overview:** The latest and most advanced model in the MiniCPM-V series, built on Qwen3-8B and SigLIP2-400M, featuring 8B parameters.
*   **Capabilities:** Outperforms leading proprietary and open-source models in vision-language tasks.  Includes new features like hybrid fast/deep thinking, efficient high-FPS and long video understanding, and strong OCR.
*   **Key Techniques:**  Unified 3D-Resampler for efficient video understanding, unified pre-training for OCR and document knowledge, and hybrid fast/deep thinking with multimodal RL.

### MiniCPM-o 2.6 (Latest)

*   **Overview:**  The most capable model in the MiniCPM-o series, built in an end-to-end fashion, featuring 8B parameters.
*   **Capabilities:** Achieves GPT-4o-level performance on vision, speech, and multimodal live streaming, supporting real-time speech conversations and multimodal live streaming.
*   **Key Features:** End-to-end speech modeling with configurable voices and voice cloning.

## Quick Start: Get Started with the Demo

*   **Online Demo:** Try out the online demo for [MiniCPM-o 2.6](https://minicpm-omni-webdemo-us.modelbest.cn/) | [MiniCPM-V 2.6](http://120.92.209.146:8887/) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)
*   **Local WebUI Demo:**  Follow the instructions in the [Chat with Our Demo on Gradio](#chat-with-our-demo-on-gradio-) section to set up a local web demo, including real-time voice/video call demo and chatbot demo.

## Inference

*   **Model Zoo:** Explore the available models and download links in the [Model Zoo](#model-zoo) section.
*   **Multi-turn Conversation:** Learn how to run multi-turn conversations in the [Multi-turn Conversation](#multi-turn-conversation) section.
*   **Advanced Inference:** Discover advanced inference techniques, including:
    *   [Inference on Multiple GPUs](#inference-on-multiple-gpus)
    *   [Inference on Mac](#inference-on-mac)
    *   [Efficient Inference with llama.cpp, Ollama, vLLM](#efficient-inference-with-llamacpp-ollama-vllm)

## Fine-tuning

*   **Simple Fine-tuning:**  Get started with basic fine-tuning using Hugging Face in the [Fine-tuning](#fine-tuning) section.
*   **Advanced Fine-tuning:**  Explore more advanced fine-tuning techniques:
    *   [With Align-Anything](#with-align-anything)
    *   [With LLaMA-Factory](#with-llama-factory)
    *   [With the SWIFT Framework](#with-the-swift-framework)

## Community & Resources

*   **MiniCPM-V & o Cookbook:** Find ready-to-deploy solutions in our structured [cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook).
*   **FAQs:** Get answers to common questions in the [FAQs](#faqs) section.
*   **Awesome work using MiniCPM-V & MiniCPM-o:** Check out the applications that use our models in the [Awesome work using MiniCPM-V & MiniCPM-o](#awesome-work-using-minicpm-v--minicpm-o) section.

## Contribution
We welcome contributions to the MiniCPM-V project.

## License

MiniCPM-o/V models are licensed under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM-V/blob/main/LICENSE) license.