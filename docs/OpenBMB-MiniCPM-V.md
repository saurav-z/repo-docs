<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" alt="MiniCPM-V & MiniCPM-o">

</div>

# MiniCPM-V and MiniCPM-o: Powerful Multimodal LLMs for On-Device Understanding

**Unlock cutting-edge multimodal capabilities on your phone with MiniCPM-V and MiniCPM-o, offering state-of-the-art performance for single image, multi-image, high-FPS video, and audio understanding.**  [Explore the original repository](https://github.com/OpenBMB/MiniCPM-V).

---

**Key Features & Highlights:**

*   **Exceptional Performance:** MiniCPM models consistently achieve top scores on leading benchmarks, *outperforming* many larger, proprietary models. MiniCPM-V 4.5 is at GPT-4o level and MiniCPM-o 2.6 is at GPT-4o level.

*   **On-Device Efficiency:** Designed for efficient deployment on mobile devices, enabling real-time multimodal processing on your phone or tablet.

*   **Versatile Input Support:** Process images, videos, text, and audio as inputs, enabling a wide range of applications.

*   **High-FPS & Long Video Understanding:** The MiniCPM-V series now supports efficient high-FPS and long video understanding with a new unified 3D-Resampler.

*   **Multimodal Live Streaming:** MiniCPM-o 2.6 supports multimodal live streaming, including video and audio streams.

*   **Multilingual Support:** Supports multilingual capabilities in over 30 languages.

*   **Easy Integration:** Supports popular frameworks like llama.cpp, Ollama, and vLLM, and can be integrated in iOS/web demo.

---

## MiniCPM-V 4.5

The latest and most capable model in the MiniCPM-V series. Built on Qwen3-8B and SigLIP2-400M with a total of 8B parameters.

**Key Improvements:**

*   **Vision-Language Leader:** Surpasses GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B on vision-language capabilities.
*   **Efficient Video Processing:** Achieves a 96x compression rate for video tokens.
*   **Controllable Thinking Modes:** Supports both fast and deep thinking modes.
*   **Enhanced OCR and Document Parsing:** Leading performance on OCRBench, and leading performance on PDF document parsing on OmniDocBench.
*   **Trustworthy and Multilingual:** Features trustworthy behaviors and supports multilingual capabilities.

### Key Techniques

*   **Unified 3D-Resampler for High-density Video Compression**
*   **Pre-training: Unified Learning for OCR and Knowledge from Documents**
*   **Post-training: Hybrid Fast/Deep Thinking with Multimodal RL.**

## MiniCPM-o 2.6

The most capable model in the MiniCPM-o series, excelling in vision, speech, and multimodal live streaming. Built in an end-to-end fashion based on SigLip-400M, Whisper-medium-300M, ChatTTS-200M, and Qwen2.5-7B with a total of 8B parameters.

**Key Highlights:**

*   **Vision and Audio Leader:** Achieves GPT-4o level performance for vision, speech, and multimodal live streaming on end-side devices.
*   **Bilingual Real-time Speech Conversation:** Supports bilingual real-time speech conversation with configurable voices.
*   **Multimodal Live Streaming Capability:** It can accept continuous video and audio streams and support real-time speech interaction.
*   **Superior Efficiency:** State-of-the-art token density for faster inference and lower memory usage.
*   **Key Techniques:**
    *   End-to-end Omni-modal Architecture
    *   Omni-modal Live Streaming Mechanism
    *   Configurable Speech Modeling Design

## MiniCPM-V & o Cookbook

Dive into ready-to-deploy solutions for the MiniCPM-V and MiniCPM-o series with our comprehensive cookbook.

**Key Features:**

*   Easy Usage Documentation
*   Broad User Spectrum
*   Versatile Deployment Scenarios

## Inference

*   **Model Zoo:** Access various model versions for different hardware and use cases.

*   **Multi-turn Conversation:** Enables conversational interactions with images and videos.

*   **Efficient Inference:** Supports llama.cpp, Ollama, and vLLM for optimal performance on various devices.

*   **Audio Understanding:** Addressed ASR, speaker analysis, general audio captioning, and sound scene tagging.

## Fine-tuning

*   **Simple Fine-tuning:** Easy fine-tuning with Hugging Face.
*   **With Align-Anything:** Fine-tuning MiniCPM-o 2.6 by PKU-Alignment Team (both vision and audio, SFT and DPO)
*   **With LLaMA-Factory:** Fine-tuning MiniCPM-o 2.6 and MiniCPM-V 2.6 with the LLaMA-Factory framework.
*   **With the SWIFT Framework:** Fine-tuning MiniCPM-V series

## Awesome Work

*   [text-extract-api](https://github.com/CatchTheTornado/text-extract-api)
*   [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party)
*   [Ollama-OCR](https://github.com/imanoop7/Ollama-OCR)
*   [comfyui-mixlab-nodes](https://github.com/MixLabPro/comfyui-mixlab-nodes)
*   [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)
*   [pensieve](https://github.com/arkohut/pensieve)
*   [paperless-gpt](https://github.com/icereed/paperless-gpt)
*   [Neuro](https://github.com/kimjammer/Neuro)

## FAQs

Find answers to common questions in the [FAQs](./docs/faqs.md) section.

## Limitations

See the [limitations section](#limitations) for details.

## Model License

*   Apache-2.0

*   MiniCPM Model License

## Statement

See the [Statement](#statement) section for details.

## Institutions

*   THUNLP
*   ModelBest

## 🌟 Star History

[Include Star History Image (e.g., using a service like api.star-history.com as shown in the original)]

## Key Techniques and Other Multimodal Projects

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLPR](https://github.com/OpenBMB/RLPR) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)

## Citation

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```