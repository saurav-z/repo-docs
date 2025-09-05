<div align="center">
  <img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" alt="MiniCPM-V & MiniCPM-o Title">
</div>

# MiniCPM-V & MiniCPM-o: Powerful Multimodal LLMs for On-Device Understanding

**MiniCPM-V & MiniCPM-o** are a series of cutting-edge, efficient multimodal Large Language Models (MLLMs) designed for superior on-device understanding. Experience GPT-4o-level performance on your phone!

[![中文](./README_zh.md)] [English]
<br>

<span style="display: inline-flex; align-items: center; margin-right: 2px;">
  <img src="./assets/wechat.png" alt="WeChat" style="margin-right: 4px;">
  <a href="docs/wechat.md" target="_blank"> WeChat</a> &nbsp;|
</span>
&nbsp;
<span style="display: inline-flex; align-items: center; margin-left: -8px;">
<img src="./assets/discord.png" alt="Discord" style="margin-right: 4px;">
  <a href="https://discord.gg/rftuRMbqzf" target="_blank"> Discord</a> &nbsp;
</span>

<p align="center">
   MiniCPM-V 4.5 <a href="https://huggingface.co/openbmb/MiniCPM-V-4_5">🤗</a> <a href="http://101.126.42.235:30910/">🤖</a> | MiniCPM-o 2.6 <a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">🤗</a>  <a href="https://minicpm-omni-webdemo-us.modelbest.cn/"> 🤖</a> | <a href="https://github.com/OpenSQZ/MiniCPM-V-Cookbook">🍳 Cookbook</a> |
  📄 Technical Report (Coming Soon)
</p>

Explore the power of MiniCPM-V & MiniCPM-o. [Visit the original repository](https://github.com/OpenBMB/MiniCPM-V) for more details.

**Key Features:**

*   **MiniCPM-V 4.5:** 🔥🔥🔥 The latest in the series, **outperforming GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B** in vision-language tasks. Features include:

    *   Efficient high-FPS and long video understanding (up to 96x compression).
    *   Controllable hybrid fast/deep thinking for enhanced performance.
    *   Robust OCR and complex document parsing capabilities.
    *   Trustworthy behavior, multilingual support, and end-side deployability.
*   **MiniCPM-o 2.6:** ⭐️⭐️⭐️ The most capable model in the MiniCPM-o series.
    *   **Achieves comparable performance to GPT-4o-202405 in vision, speech, and multimodal live streaming.**
    *   Bilingual real-time speech conversation with customizable voices.
    *   Supports multimodal live streaming on end-side devices like the iPad.

**Contents:**

*   [MiniCPM-V 4.5](#minicpm-v-45)
    *   [Key Techniques](#key-techniques)
    *   [Inference Efficiency](#inference-efficiency)
    *   [Examples](#examples)
*   [MiniCPM-o 2.6](#minicpm-o-26)
    *   [Examples](#examples)
*   [Legacy Models](#legacy-models)
*   [MiniCPM-V & o Cookbook](#minicpm-v--o-cookbook)
*   [Chat with Our Demo on Gradio 🤗](#chat-with-our-demo-on-gradio-)
    *   [Online Demo](#online-demo)
    *   [Local WebUI Demo](#local-webui-demo)
*   [Inference](#inference)
    *   [Model Zoo](#model-zoo)
    *   [Multi-turn Conversation](#multi-turn-conversation)
        *   [Chat with Multiple Images](#chat-with-multiple-images)
        *   [In-context Few-shot Learning](#in-context-few-shot-learning)
        *   [Chat with Video](#chat-with-video)
        *   [Speech and Audio Mode](#speech-and-audio-mode)
            *   [Mimick](#mimick)
            *   [General Speech Conversation with Configurable Voices](#general-speech-conversation-with-configurable-voices)
            *   [Speech Conversation as an AI Assistant](#speech-conversation-as-an-ai-assistant)
            *   [Instruction-to-Speech](#instruction-to-speech)
            *   [Voice Cloning](#voice-cloning)
            *   [Addressing Various Audio Understanding Tasks](#addressing-various-audio-understanding-tasks)
        *   [Multimodal Live Streaming](#multimodal-live-streaming)
    *   [Inference on Multiple GPUs](#inference-on-multiple-gpus)
    *   [Inference on Mac](#inference-on-mac)
    *   [Efficient Inference with llama.cpp, Ollama, vLLM](#efficient-inference-with-llamacpp-ollama-vllm)
*   [Fine-tuning](#fine-tuning)
    *   [Simple Fine-tuning](#simple-fine-tuning)
    *   [With Align-Anything](#with-align-anything)
    *   [With LLaMA-Factory](#with-llama-factory)
    *   [With the SWIFT Framework](#with-the-swift-framework)
*   [Awesome work using MiniCPM-V & MiniCPM-o](#awesome-work-using-minicpm-v--minicpm-o)
*   [FAQs](#faqs)
*   [Limitations](#limitations)
*   [Model License](#model-license)
*   [Statement](#statement)
*   [Institutions](#institutions)
*   [🌟 Star History](#star-history)
*   [Key Techniques and Other Multimodal Projects](#key-techniques-and-other-multimodal-projects)
*   [Citation](#citation)

## MiniCPM-V 4.5

The latest MiniCPM-V, built on Qwen3-8B and SigLIP2-400M, offers state-of-the-art performance.

**Key Highlights:**

*   **Superior Vision-Language Capability:** Surpasses GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B with just 8B parameters.
*   **Efficient Video Understanding:** Enables high-FPS and long video understanding with a 96x compression rate for video tokens.
*   **Controllable Reasoning:** Supports both fast and deep thinking modes for optimized performance.
*   **Enhanced OCR and Document Processing:** Delivers top-tier performance on OCRBench and OmniDocBench.
*   **Easy to Use:** Supports llama.cpp, Ollama, vLLM, int4/GGUF/AWQ quantization, SGLang, fine-tuning, local WebUI demos, and iOS apps.

### Key Techniques
<div align="center">
<img src="./assets/minicpm-v-4dot5-framework.png" , width=100%>
</div>
* **Unified 3D-Resampler for High-density Video Compression.**
* **Pre-training: Unified Learning for OCR and Knowledge from Documents.**
* **Post-training: Hybrid Fast/Deep Thinking with Multimodal RL.**

### Evaluation

<div align="center">
  <img src="./assets/radar_minicpm_v45.png", width=60%>
</div>

<div align="center">
<img src="./assets/minicpmv_4_5_evaluation_result.png" , width=80%>
</div>

### Inference Efficiency

**OpenCompass**
<div align="left">
<table style="margin: 0px auto;">
    <thead>
            <tr>
              <th align="left">Model</th>
              <th>Size</th>
              <th>Avg Score ↑</th>
              <th>Total Inference Time ↓</th>
            </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td nowrap="nowrap" align="left">GLM-4.1V-9B-Thinking</td>
            <td>10.3B</td>
            <td>76.6</td>
            <td>17.5h</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiMo-VL-7B-RL</td>
            <td>8.3B</td>
            <td>76.4</td>
            <td>11h</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiniCPM-V 4.5</td>
            <td>8.7B</td>
            <td><b>77.0</td>
            <td><b>7.5h</td>
        </tr>
    </tbody>
</table>
</div>

**Video-MME**

<div align="left">
<table style="margin: 0px auto;">
    <thead>
          <tr>
              <th align="left">Model</th>
              <th>Size</th>
              <th>Avg Score ↑</th>
              <th>Total Inference Time ↓</th>
              <th>GPU Mem ↓</th>
          </tr>
    </thead>
    <tbody align="center">
          <tr>
              <td nowrap="nowrap" align="left">Qwen2.5-VL-7B-Instruct</td>
              <td>8.3B</td>
              <td>71.6</td>
              <td>3h</td>
              <td>60G</td>
          </tr>
          <tr>
              <td nowrap="nowrap" align="left">GLM-4.1V-9B-Thinking</td>
              <td>10.3B</td>
              <td><b>73.6</b></td>
              <td>2.63h</td>
              <td>32G</td>
          </tr>
          <tr>
              <td nowrap="nowrap" align="left">MiniCPM-V 4.5</td>
              <td>8.7B</td>
              <td>73.5</td>
              <td><b>0.26h</b></td>
              <td><b>28G</b></td>
        </tr>
    </tbody>
</table>
</div>

### Examples
<div align="center">
  <a href="https://www.youtube.com/watch?v=Cn23FujYMMU"><img src="./assets/minicpmv4_5/MiniCPM-V 4.5-8.26_img.jpeg", width=70%></a>
</div>

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmv4_5/en_case1.png" alt="en_case1" style="margin-bottom: 5px;">
  <img src="assets/minicpmv4_5/en_case2.png" alt="en_case2" style="margin-bottom: 5px;">
  <img src="assets/minicpmv4_5/en_case3.jpeg" alt="en_case3" style="margin-bottom: 5px;">
</div>

<details>
<summary>Click to view more cases.</summary>
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmv4_5/zh_extra.jpeg" alt="zh_extra" style="margin-bottom: 5px;">
</div>
</details>

## MiniCPM-o 2.6

The latest MiniCPM-o model, built end-to-end, achieves impressive results in vision, speech, and multimodal live streaming.

**Key Features:**

*   **Exceptional Visual Understanding:** Outperforms GPT-4o-202405 and Gemini 1.5 Pro.
*   **Leading Speech Capability:** Bilingual real-time speech conversation with customizable voices.
*   **Multimodal Live Streaming:** Supports continuous video and audio streams with real-time speech interaction.
*   **High Efficiency:** Offers state-of-the-art token density, improving speed and resource usage.

**Model Architecture.**
- **End-to-end Omni-modal Architecture.**
- **Omni-modal Live Streaming Mechanism.**
- **Configurable Speech Modeling Design.**

<div align="center">
<img src="./assets/minicpm-o-26-framework-v2.png" , width=80%>
</div>

### Evaluation
<div align="center">
  <img src="./assets/radar.jpg", width=80%>
</div>

### Examples
<div align="center">
  <a href="https://www.youtube.com/watch?v=vRIMbxJzStY&t=2s"><img src="./assets/minicpmo2_6/2dot6_o_demo_video_img.png", width=70%></a>
</div>

<br>

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmo2_6/minicpmo2_6_math_intersect.png" alt="math" style="margin-bottom: 5px;">
  <img src="assets/minicpmo2_6/minicpmo2_6_diagram_train_NN.png" alt="diagram" style="margin-bottom: 5px;">
  <img src="assets/minicpmo2_6/minicpmo2_6_multi-image_bike.png" alt="bike" style="margin-bottom: 5px;">
</div>

## Legacy Models
  
| Model                | Introduction and Guidance       |
|:----------------------|:-------------------:|
| MiniCPM-V 4.0  | [Document](./docs/minicpm_v4_en.md)   | 
| MiniCPM-V 2.6  | [Document](./docs/minicpm_v2dot6_en.md)   | 
| MiniCPM-Llama3-V 2.5  | [Document](./docs/minicpm_llama3_v2dot5.md)   | 
| MiniCPM-V 2.0  | [Document](./docs/minicpm_v2.md)   | 
| MiniCPM-V 1.0  | [Document](./docs/minicpm_v1.md)   | 
| OmniLMM-12B  | [Document](././docs/omnilmm_en.md)   |  

## MiniCPM-V & o Cookbook

Explore comprehensive, ready-to-deploy solutions for the MiniCPM-V and MiniCPM-o model series.

**Key Features:**

*   **Easy Usage Documentation**
*   **Broad User Spectrum**: Individuals, Enterprises, and Researchers supported.
*   **Versatile Deployment Scenarios**: Web demo, Quantized deployment, End devices.

## Chat with Our Demo on Gradio 🤗

Experience MiniCPM-V and MiniCPM-o with user-friendly Gradio demos.

### Online Demo
*   [MiniCPM-o 2.6](https://minicpm-omni-webdemo-us.modelbest.cn/) | [MiniCPM-V 2.6](http://120.92.209.146:8887/) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)

### Local WebUI Demo

Create your own local WebUI demo.

## Inference

### Model Zoo

| Model           | Device | Memory    | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Description       | Download |
|:-----------|:--:|:-----------:|:-------------------|:---------------:|
| MiniCPM-V 4.5| GPU | 18 GB  | The latest version, strong end-side multimodal performance for single image, multi-image and video understanding.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5) |
| MiniCPM-V 4.5 gguf | CPU | 8 GB  | The gguf version, lower memory usage and faster inference.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-gguf) |
| MiniCPM-V 4.5 int4 | GPU | 9 GB  | The int4 quantized version, lower GPU memory usage.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4) |
| MiniCPM-V 4.5 AWQ | GPU | 9 GB  | The int4 quantized version, lower GPU memory usage.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-AWQ) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-AWQ) |
| MiniCPM-o 2.6| GPU | 18 GB  | The latest version, achieving GPT-4o level performance for vision, speech and multimodal live streaming on end-side devices.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6) |
| MiniCPM-o 2.6 gguf | CPU | 8 GB  | The gguf version, lower memory usage and faster inference.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-gguf) |
| MiniCPM-o 2.6 int4 | GPU | 9 GB  | The int4 quantized version, lower GPU memory usage.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-int4) |

### Multi-turn Conversation

Run multi-turn conversations with the provided code samples.

### Inference on Multiple GPUs

Distribute layers across multiple GPUs for low VRAM. See the [tutorial](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md).

### Inference on Mac

Run MiniCPM-Llama3-V 2.5 on Mac with MPS using the provided example.

### Efficient Inference with llama.cpp, Ollama, vLLM

Utilize llama.cpp, Ollama, and vLLM for efficient inference.

## Fine-tuning

Fine-tune MiniCPM-V and MiniCPM-o models with various frameworks.

### Simple Fine-tuning
### With Align-Anything
### With LLaMA-Factory
### With the SWIFT Framework

## Awesome work using MiniCPM-V & MiniCPM-o

Explore projects built using MiniCPM-V & MiniCPM-o.

## FAQs

Find answers to frequently asked questions [here](./docs/faqs.md).

## Limitations

Review the limitations of MiniCPM-o 2.6.

## Model License

*   This repository is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License.

*   The usage of MiniCPM-o/V model weights must strictly follow [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).

*   The models and weights of MiniCPM are completely free for academic research. after filling out a ["questionnaire"](https://modelbest.feishu.cn/share/base/form/shrcnpV5ZT9EJ6xYjh3Kx0J6v8g) for registration, are also available for free commercial use.

## Statement

Learn more about content generation and model usage.

## Institutions

This project is developed by:

*   <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
*   <img src="assets/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)

## 🌟 Star History

<table align="center">
    <p align="center">
      <img src="assets/star-history-25-09-02.png"/>
    </p>
</table>

## Key Techniques and Other Multimodal Projects
👏 Welcome to explore key techniques of MiniCPM-o/V and other multimodal projects of our team:

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLPR](https://github.com/OpenBMB/RLPR) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)

## Citation

Cite our work if you found it helpful!
```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```