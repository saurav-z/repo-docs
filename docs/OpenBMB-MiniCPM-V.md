<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

</div>

# MiniCPM-V & MiniCPM-o: Powerful Multimodal LLMs for On-Device Understanding

**Unlock cutting-edge multimodal understanding on your device with MiniCPM-V and MiniCPM-o, achieving GPT-4o-level performance with efficiency.** 

[View the original repository](https://github.com/OpenBMB/MiniCPM-V-4_5)

## Key Features

*   ✅ **State-of-the-Art Performance:** MiniCPM-V 4.5 and MiniCPM-o 2.6 models deliver exceptional performance, rivaling or surpassing leading models like GPT-4o, Gemini, and Claude, while being optimized for efficient deployment.
*   🎬 **High-FPS and Long Video Understanding:** MiniCPM-V 4.5 processes video at high speeds and can understand long videos using the innovative 3D-Resampler for efficient token usage.
*   🗣️ **Advanced Speech & Audio Capabilities:** MiniCPM-o 2.6 offers multilingual real-time speech conversation, voice cloning, and customizable voice control for a rich audio experience.
*   🖼️ **Versatile Multimodal Support:**  Process images, videos, and text as inputs and generate high-quality text outputs. MiniCPM-o also supports audio.
*   📱 **On-Device Deployment:**  Designed for efficient end-side deployment, enabling use on mobile devices like phones and tablets.
*   🚀 **Flexible and Easy-to-Use:** Integrate with llama.cpp, Ollama, and vLLM for inference, and offers options for fine-tuning and custom integration.
*   📚 **Comprehensive Resource:** Includes a MiniCPM-V & o Cookbook, online and local demos, and detailed documentation.

## Model Highlights

### MiniCPM-V 4.5

The latest in the MiniCPM-V series, built on Qwen3-8B and SigLIP2-400M with a total of 8B parameters, offering a variety of features including:

*   **Superior Vision-Language Capabilities**: Outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B, making it the most performant MLLM under 30B parameters for vision-language tasks.
*   **Efficient Video Processing**: Utilizing 3D-Resampler for high-FPS and long video understanding with compression rates up to 96x for video tokens.
*   **Controllable Hybrid Thinking**: Supports both "fast" and "deep" thinking modes for flexibility in various usage scenarios.
*   **Enhanced OCR and Document Parsing**: Improved OCR capabilities and parsing of PDF documents.
*   **Easy to Use**: Ready to be deployed via llama.cpp, ollama, vLLM, and other frameworks, with options for int4 and GGUF quantized models.

### MiniCPM-o 2.6

The latest in the MiniCPM-o series, offers state-of-the-art performance, built in an end-to-end fashion based on SigLip-400M, Whisper-medium-300M, ChatTTS-200M, and Qwen2.5-7B with a total of 8B parameters. Key features include:

*   **Excellent Visual Understanding**: Outperforms GPT-4o-202405, Gemini 1.5 Pro, and Claude 3.5 Sonnet in single-image understanding, with improvements in multi-image and video understanding capabilities.
*   **Advanced Speech Generation**: The model supports bilingual real-time speech conversation, voice cloning, and customizable voice control.
*   **High-Quality, Real-time Multimodal Streaming**: Accepts continuous video and audio streams, supporting real-time speech interaction and outperforming GPT-4o-202408 and Claude 3.5 Sonnet on StreamingBench.
*   **Easy to Use**: Can be integrated via llama.cpp, vLLM, and is available in quantized formats for efficiency.

## Quick Start

*   **Model Zoo:** Explore the available models and download the one that best suits your needs.
*   **Multi-Turn Conversation:** Use the provided code examples to have multi-turn conversations with images and videos.
*   **Inference:** Integrate the model with tools like llama.cpp, Ollama, vLLM, and others.
*   **Fine-tuning:** Adapt the model to specific tasks and datasets using the provided fine-tuning instructions.
*   **Demos:** Use the online and local demos to explore and test the capabilities of the models.

## Key Technologies and Related Projects

Explore the core technologies and related projects:

*   [VisCPM](https://github.com/OpenBMB/VisCPM/tree/main)
*   [RLPR](https://github.com/OpenBMB/RLPR)
*   [RLHF-V](https://github.com/RLHF-V/RLHF-V)
*   [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD)
*   [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)

## Resources

*   **MiniCPM-V & o Cookbook:** Get ready-to-use solutions and guidance for implementing multimodal AI applications.
    *   [Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)
*   **Online Demos:**  Experience MiniCPM-o 2.6 and MiniCPM-V 2.0 directly.
    *   [MiniCPM-o 2.6 Demo](https://minicpm-omni-webdemo-us.modelbest.cn/)
    *   [MiniCPM-V 2.0 Demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)
*   **FAQs:** Find answers to common questions.
    *   [FAQs](./docs/faqs.md)

## Citation
```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}