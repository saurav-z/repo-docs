<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

</div>

# MiniCPM-V & MiniCPM-o: Cutting-Edge Multimodal AI for On-Device Applications

**MiniCPM-V and MiniCPM-o** are efficient, state-of-the-art multimodal large language models (MLLMs) designed for powerful AI experiences on your phone and other end-side devices. Offering exceptional vision, speech, and multimodal capabilities, these models empower users with high-quality text and speech outputs.

[**View the original repository on GitHub**](https://github.com/OpenBMB/MiniCPM-V)

**Key Features of MiniCPM-V & MiniCPM-o:**

*   **Exceptional Performance:** Achieve GPT-4o-level performance on on-device multimodal tasks.
*   **Multimodal Input:** Supports images, videos, text, and audio.
*   **End-to-End Capability:** MiniCPM-o offers end-to-end capabilities including speech generation and speech conversation.
*   **Efficient Deployment:** Designed for efficient, on-device deployment, offering high-FPS and long video understanding.
*   **Flexible Usage:** Supported by llama.cpp, Ollama, and vLLM.
*   **Comprehensive Toolkit**: Leverage the MiniCPM-V & o Cookbook for easy deployment.

## MiniCPM-V 4.5:  The Most Capable Open-Source MLLM

MiniCPM-V 4.5 is the latest and most capable model in the MiniCPM-V series. Built on Qwen3-8B and SigLIP2-400M with a total of 8B parameters, it surpasses other existing models in vision-language capabilities, making it the most performant MLLM under 30B parameters.

**Key Highlights:**

*   **Unrivaled Vision-Language Ability:** Performs better than GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language capabilities.
*   **High-Performance Video Understanding:** Featuring efficient high-FPS (up to 10FPS) and long video understanding (up to 96x compression rate for video tokens).
*   **Versatile Capabilities:** The model supports Controllable hybrid fast/deep thinking, strong OCR, document parsing and multilingual support.
*   **Ease of Use:** Compatible with various frameworks including llama.cpp, Ollama, vLLM, and SGLang. Offers quantized model formats (int4, GGUF, AWQ) and supports fine-tuning.
*   **Online Demo:**  [Try out the online demo](http://101.126.42.235:30910/)

## MiniCPM-o 2.6:  GPT-4o Level Speech and Multimodal Live Streaming

MiniCPM-o 2.6 is the most capable model in the MiniCPM-o series. It's built in an end-to-end fashion based on SigLip-400M, Whisper-medium-300M, ChatTTS-200M, and Qwen2.5-7B with a total of 8B parameters. It exhibits a significant performance improvement over MiniCPM-V 2.6, and introduces new features for real-time speech conversation and multimodal live streaming.

**Key Highlights:**

*   **GPT-4o-Level Performance:** Performs similar to GPT-4o-202405.
*   **Exceptional Speech Capability:** Supports bilingual real-time speech conversation with configurable voices and provides voice cloning.
*   **State-of-the-art Multimodal Live Streaming:** Accepts continuous video and audio streams.
*   **Superior Efficiency:**  Supports multimodal live streaming on devices like iPads.
*   **Online Demo:**  [Try out the online demo](https://minicpm-omni-webdemo-us.modelbest.cn/)

## MiniCPM-V & o Cookbook

The MiniCPM-V & o Cookbook provides:
- Easy Usage Documentation
- Broad User Spectrum
- Versatile Deployment Scenarios

Access the cookbook here: [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)

## Demos: Experience MiniCPM-V & MiniCPM-o

*   [MiniCPM-o 2.6 Online Demo](https://minicpm-omni-webdemo-us.modelbest.cn/)
*   [MiniCPM-V 2.6 Online Demo](http://120.92.209.146:8887/)
*   [MiniCPM-Llama3-V 2.5 Online Demo](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5)
*   [MiniCPM-V 2.0 Online Demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)

## Inference: Run the Models

*   **Model Zoo:** Access the latest models and quantized versions.
*   **Multi-turn Conversation:**  Demonstration code for multi-turn interactions, including with images, video, and audio.
*   **Efficient Inference:** Instructions for using llama.cpp, Ollama, and vLLM for efficient inference.

## Fine-tuning: Customize for Your Needs

*   **Simple Fine-tuning:** Easy fine-tuning with Hugging Face.
*   **Advanced Frameworks:** Support for fine-tuning with Align-Anything, LLaMA-Factory, and SWIFT.

## Awesome Projects Using MiniCPM-V & MiniCPM-o

Explore projects built on MiniCPM-V and MiniCPM-o, including OCR APIs, ComfyUI integrations, and more.

## FAQs

Find answers to common questions in our [FAQs](./docs/faqs.md).

## Limitations

See the limitations of these models at the [Limitations Section](#limitations).

## Model License

This project is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License.
Please adhere to [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).
Free for academic research, and free for commercial use after a registration.

## Statement

The use of the MiniCPM-o/V models is at your own risk.
See the [Statement Section](#statement) for more information.

## Institutions

This project is developed by:

*   [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
*   [ModelBest](https://modelbest.cn/)

## Star History

```html
<table align="center">
    <p align="center">
      <img src="assets/star-history-25-09-02.png"/>
    </p>
</table>
```

## Key Techniques and Other Multimodal Projects

Explore the key techniques and related projects from our team:

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLPR](https://github.com/OpenBMB/RLPR) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)

## Citation

Cite our research if you find our models/code/paper helpful:

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```