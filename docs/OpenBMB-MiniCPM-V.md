<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

</div>

# MiniCPM-V & MiniCPM-o: Cutting-Edge Multimodal LLMs for On-Device Intelligence

**MiniCPM-V & MiniCPM-o** offer powerful, efficient, and versatile multimodal language models (MLLMs) for understanding images, videos, audio, and text. [Explore the original repository](https://github.com/OpenBMB/MiniCPM-V) for more details.

## Key Features

*   **On-Device Performance:** Optimized for efficient deployment on phones, tablets, and other edge devices.
*   **Multimodal Input:** Process images, videos, audio, and text for rich context understanding.
*   **High-Quality Output:** Generate human-like text responses and, in the case of MiniCPM-o, speech output.
*   **Open Source & Accessible:** Free for academic research and commercial use, with an easy-to-use cookbook and demos.

### MiniCPM-V 4.5: The Pinnacle of Vision-Language Understanding

MiniCPM-V 4.5 is the latest iteration, setting a new standard in the open-source community.

*   **Top-Tier Performance:** Outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language tasks.
*   **Advanced Video Understanding:** Achieves efficient high-FPS and long video understanding.
*   **Key Techniques:** Unified 3D-Resampler and Pre-training for unified OCR and knowledge from documents.
*   **Enhanced Functionality:** Controllable hybrid fast/deep thinking and multilingual support.

### MiniCPM-o 2.6: The All-in-One Multimodal Experience

MiniCPM-o 2.6 is a leading end-to-end model, bringing together vision, speech, and live streaming.

*   **Leading Performance:** Achieves comparable performance to GPT-4o-202405 in vision, speech, and multimodal live streaming.
*   **Key Techniques:** End-to-end Omni-modal Architecture and Configurable Speech Modeling Design.
*   **Real-time Speech & Audio:** Supports bilingual real-time speech conversation and voice cloning.
*   **Multimodal Streaming:** Enables live video and audio stream understanding on end-side devices.

## Quick Links

*   [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook): Comprehensive guides for diverse user scenarios
*   [Online Demos](https://minicpm-omni-webdemo-us.modelbest.cn/) & [Local WebUI Demo](#local-webui-demo): Try our latest models
*   [Model Zoo](#model-zoo): Access pre-trained models
*   [FAQs](#faqs): Get answers to common questions

## News

*   **[2025.08.26]** 🔥🔥🔥 We open-source MiniCPM-V 4.5, which outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B. It advances popular capabilities of MiniCPM-V, and brings useful new features. Try it now!
*   **[2025.09.01]** ⭐️⭐️⭐️ MiniCPM-V 4.5 has been officially supported by [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15575), [vLLM](https://github.com/vllm-project/vllm/pull/23586), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/pull/9022).
*   **[2025.08.01]** ⭐️⭐️⭐️ We open-sourced the [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)! It provides comprehensive guides for diverse user scenarios, paired with our new [Docs Site](https://minicpm-o.readthedocs.io/en/latest/index.html) for smoother onboarding.
*   **More News** See [README](https://github.com/OpenBMB/MiniCPM-V).

## Model Usage & Inference

### Model Zoo

Access and download pre-trained models for various use cases:

| Model           | Device | Memory    | Description                                                                                             | Download                                                                                                                                                                                                        |
| :-------------- | :----- | :-------- | :------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MiniCPM-V 4.5   | GPU    | 18 GB     | The latest version, strong end-side multimodal performance for single image, multi-image and video understanding. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5)                      |
| MiniCPM-V 4.5 gguf  | CPU    | 8 GB  | The gguf version, lower memory usage and faster inference.   | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-gguf) |
| MiniCPM-V 4.5 int4 | GPU    | 9 GB     | The int4 quantized version, lower GPU memory usage.                                                   | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4)                        |
| MiniCPM-V 4.5 AWQ | GPU    | 9 GB     | The int4 quantized version, lower GPU memory usage.                                                   | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-AWQ) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://huggingface.co/openbmb/MiniCPM-V-4_5-AWQ)                         |
| MiniCPM-o 2.6   | GPU    | 18 GB     | The latest version, achieving GPT-4o level performance for vision, speech and multimodal live streaming on end-side devices.    | [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6)                          |
| MiniCPM-o 2.6 gguf  | CPU    | 8 GB  | The gguf version, lower memory usage and faster inference.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-gguf) |
| MiniCPM-o 2.6 int4 | GPU    | 9 GB     | The int4 quantized version, lower GPU memory usage.                                                   | [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-int4)                          |

### Multi-turn Conversation

Follow the code provided in [Multi-turn Conversation](#multi-turn-conversation) to start a conversation. You can also refer to [Chat with Multiple Images](#chat-with-multiple-images), [In-context Few-shot Learning](#in-context-few-shot-learning), [Chat with Video](#chat-with-video) to get more information.

### Speech and Audio Mode

You can refer to [Speech and Audio Mode](#speech-and-audio-mode) to create your own speech or audio.

### Multimodal Live Streaming

You can refer to [Multimodal Live Streaming](#multimodal-live-streaming) to get more information about Live Streaming.

### Efficient Inference with llama.cpp, Ollama, vLLM

*   [llama.cpp](https://github.com/OpenBMB/llama.cpp/tree/minicpmv-main/examples/llava/README-minicpmv2.6.md)
*   [Ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md)
*   [vLLM](https://docs.vllm.ai/en/latest/getting_started/examples/vision_language.html)

## Fine-tuning

*   [Simple Fine-tuning](./finetune/readme.md)
*   [With Align-Anything](https://github.com/PKU-Alignment/align-anything/tree/main/scripts).
*   [With LLaMA-Factory](./docs/llamafactory_train_and_infer.md)
*   [With the SWIFT Framework](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md)

## FAQs

*   See the [FAQs](./docs/faqs.md)

## Limitations

*   [Limitations](#limitations)

## Citation

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```