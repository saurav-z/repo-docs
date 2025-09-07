<div align="center">
  <img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" alt="MiniCPM-V & MiniCPM-o Title">
</div>

# MiniCPM-V: Unleash GPT-4o-Level Multimodal AI on Your Device

**MiniCPM-V** is a powerful series of efficient, on-device multimodal LLMs (MLLMs) that bring GPT-4o-level performance to your phone, supporting single images, multiple images, and high-FPS video understanding.  Access the original repo [here](https://github.com/OpenBMB/MiniCPM-V).

**Key Features:**

*   **Multimodal Input:** Accepts images, videos, and text as input.  **MiniCPM-o** additionally processes audio.
*   **High-Quality Output:** Generates high-quality text and, in the case of MiniCPM-o, speech outputs.
*   **On-Device Deployment:** Designed for efficient, on-device performance.
*   **Strong Performance:** Latest MiniCPM-V 4.5 outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language capabilities. MiniCPM-o 2.6 offers comparable performance to GPT-4o-202405 in vision, speech, and multimodal live streaming
*   **Efficient Video Understanding:** MiniCPM-V 4.5 boasts efficient high-FPS and long-video understanding, with a 96x compression rate for video tokens.
*   **Versatile Functionality:** Supports multilingual capabilities, trustworthy behavior, OCR, document parsing, voice cloning, and multimodal live streaming.
*   **Ease of Use:** Deployable across various frameworks like llama.cpp, vLLM, and Ollama, with support for quantization and fine-tuning.

## Key Technologies
*   **Architechture: Unified 3D-Resampler for High-density Video Compression.** MiniCPM-V 4.5 introduces a 3D-Resampler that overcomes the performance-efficiency trade-off in video understanding. By grouping and jointly compressing up to 6 consecutive video frames into just 64 tokens (the same token count used for a single image in MiniCPM-V series), MiniCPM-V 4.5 achieves a 96× compression rate for video tokens. This allows the model to process more video frames without additional LLM computational cost, enabling high-FPS video and long video understanding. The architecture supports unified encoding for images, multi-image inputs, and videos, ensuring seamless capability and knowledge transfer.

*   **Pre-training: Unified Learning for OCR and Knowledge from Documents.** Existing MLLMs learn OCR capability and knowledge from documents in isolated training approaches. We observe that the essential difference between these two training approaches is the visibility of the text in images. By dynamically corrupting text regions in documents with varying noise levels and asking the model to reconstruct the text, the model learns to adaptively and properly switch between accurate text recognition (when text is visible) and multimodal context-based knowledge reasoning (when text is heavily obscured). This eliminates reliance on error-prone document parsers in knowledge learning from documents, and prevents hallucinations from over-augmented OCR data, resulting in top-tier OCR and multimodal knowledge performance with minimal engineering overhead.

*   **Post-training: Hybrid Fast/Deep Thinking with Multimodal RL.** MiniCPM-V 4.5 offers a balanced reasoning experience through two switchable modes: fast thinking for efficient daily use and deep thinking for complex tasks. Using a new hybrid reinforcement learning method, the model jointly optimizes both modes, significantly enhancing fast-mode performance without compromising deep-mode capability. Incorporated with [RLPR](https://github.com/OpenBMB/RLPR) and [RLAIF-V](https://github.com/RLHF-V/RLAIF-V/), it generalizes robust reasoning skills from broad multimodal data while effectively reducing hallucinations.

## Models

*   **MiniCPM-V 4.5:** The latest and most capable model in the MiniCPM-V series, delivering state-of-the-art vision-language capabilities.
*   **MiniCPM-o 2.6:** The latest and most capable model in the MiniCPM-o series. Achieveing GPT-4o level performance in vision, speech, and multimodal live streaming.

## News

*   [2025.09.01] ⭐️⭐️⭐️ MiniCPM-V 4.5 is now supported by [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15575), [vLLM](https://github.com/vllm-project/vllm/pull/23586), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/pull/9022). You are welcome to use it directly through these official channels! Support for additional frameworks such as [Ollama](https://github.com/ollama/ollama/pull/12078) and [SGLang](https://github.com/sgl-project/sglang/pull/9610) is actively in progress.
*   [2025.08.26] 🔥🔥🔥 We open-source MiniCPM-V 4.5, which outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B. It advances popular capabilities of MiniCPM-V, and brings useful new features. Try it now!
*   [2025.08.01] ⭐️⭐️⭐️ We open-sourced the [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)! It provides comprehensive guides for diverse user scenarios, paired with our new [Docs Site](https://minicpm-o.readthedocs.io/en/latest/index.html) for smoother onboarding.
*   [2025.06.20] ⭐️⭐️⭐️ Our official [Ollama repository](https://ollama.com/openbmb) is released. Try our latest models with [one click](https://ollama.com/openbmb/minicpm-o2.6)！
*   [2025.03.01] 🚀🚀🚀 RLAIF-V, the alignment technique of MiniCPM-o, is accepted by CVPR 2025 Highlights！The [code](https://github.com/RLHF-V/RLAIF-V), [dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset), [paper](https://arxiv.org/abs/2405.17220) are open-sourced!
*   [2025.01.24] 📢📢📢 MiniCPM-o 2.6 technical report is released! See [here](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9).
*   [2025.01.19] 📢 **ATTENTION!** We are currently working on merging MiniCPM-o 2.6 into the official repositories of llama.cpp, Ollama, and vllm. Until the merge is complete, please USE OUR LOCAL FORKS of [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md), [Ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md), and [vllm](https://github.com/OpenBMB/MiniCPM-o?tab=readme-ov-file#efficient-inference-with-llamacpp-ollama-vllm). **Using the official repositories before the merge may lead to unexpected issues**.
*   [2025.01.19] ⭐️⭐️⭐️ MiniCPM-o tops GitHub Trending and reaches top-2 on Hugging Face Trending!
*   [2025.01.17] We have updated the usage of MiniCPM-o 2.6 int4 quantization version and resolved the model initialization error. Click [here](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) and try it now!
*   [2025.01.13] 🔥🔥🔥 We open-source MiniCPM-o 2.6, which matches GPT-4o-202405 on vision, speech and multimodal live streaming. It advances popular capabilities of MiniCPM-V 2.6, and supports various new fun features. Try it now!
*   [2024.08.17] 🚀🚀🚀 MiniCPM-V 2.6 is now fully supported by [official](https://github.com/ggerganov/llama.cpp) llama.cpp! GGUF models of various sizes are available [here](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf).
*   [2024.08.06] 🔥🔥🔥 We open-source MiniCPM-V 2.6, which outperforms GPT-4V on single image, multi-image and video understanding. It advances popular features of MiniCPM-Llama3-V 2.5, and can support real-time video understanding on iPad. Try it now!
*   [2024.08.03] MiniCPM-Llama3-V 2.5 technical report is released! See [here](https://arxiv.org/abs/2408.01800).
*   [2024.05.23] 🔥🔥🔥 MiniCPM-V tops GitHub Trending and Hugging Face Trending! Our demo, recommended by Hugging Face Gradio’s official account, is available [here](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5). Come and try it out!

## Demo

*   **Online Demo:** Experience MiniCPM-o 2.6, MiniCPM-V 2.6, MiniCPM-Llama3-V 2.5, and MiniCPM-V 2.0 through our online demos.
*   **Local WebUI Demo:** Build your own local WebUI demos with easy-to-follow instructions and sample code.

## Inference & Usage

*   **Model Zoo:** Access the full range of MiniCPM-V and MiniCPM-o models, including quantized versions and GGUF files for optimal performance on various devices.
*   **Multi-turn Conversation:** Engage in multi-turn conversations with the models, including support for multiple images, in-context few-shot learning, video input, and speech/audio mode.
*   **Efficient Inference:** Learn how to efficiently deploy and run MiniCPM-V models with llama.cpp, Ollama, and vLLM.
*   **Fine-tuning:** Adapt the models to your specific needs with simple fine-tuning guides.

## FAQs

*   Find answers to frequently asked questions [here](./docs/faqs.md).

## Limitations

*   Speech output may be unstable.
*   Model may repeat responses.
*   High latency on overseas web demos.

## License

*   [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)
*   [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md)

## Acknowledgements

This project is a collaboration between [THUNLP](https://nlp.csai.tsinghua.edu.cn/) and [ModelBest](https://modelbest.cn/).

**Please refer to the original repository for detailed information, code, and models.**