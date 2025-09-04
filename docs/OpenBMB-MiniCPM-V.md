<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

</div>

# MiniCPM-V & MiniCPM-o: Powerful, Efficient, On-Device Multimodal LLMs

**Unleash GPT-4o-level vision, speech, and multimodal understanding on your phone with MiniCPM-V and MiniCPM-o!**

[View the original repository on GitHub](https://github.com/OpenBMB/MiniCPM-V)

## Key Features

*   **MiniCPM-V 4.5: Cutting-Edge Vision-Language Capabilities**
    *   Outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language tasks.
    *   Efficient high-FPS and long video understanding (up to 96x compression rate).
    *   Controllable hybrid fast/deep thinking for varied use cases.
    *   Robust OCR, document parsing, multilingual support, and trustworthy behavior.
    *   Easy deployment across multiple platforms (llama.cpp, Ollama, vLLM, SGLang, iOS app, etc.).
*   **MiniCPM-o 2.6: End-to-End Multimodal Excellence**
    *   Achieves GPT-4o-202405 performance in vision, speech, and multimodal live streaming.
    *   Supports bilingual real-time speech conversation with customizable voices.
    *   Enables end-to-end voice cloning and creative voice generation.
    *   Supports multimodal live streaming on end-side devices (e.g., iPad).
    *   Superior token density for faster inference, lower memory usage, and power efficiency.

## What's New

*   **MiniCPM-V 4.5 Released!** Outperforms GPT-4o and other state-of-the-art models, with optimized video understanding and more. [Try it now!](https://huggingface.co/openbmb/MiniCPM-V-4_5)
*   **MiniCPM-o 2.6 Released!** Experience GPT-4o-level performance in vision, speech, and live streaming. [Explore now!](https://huggingface.co/openbmb/MiniCPM-o-2_6)
*   **Cookbook Available!**  Comprehensive guides for various user scenarios, paired with our new [Docs Site](https://minicpm-o.readthedocs.io/en/latest/index.html) for smoother onboarding.
*   **Official Ollama Support:** Try our latest models with [one click](https://ollama.com/openbmb/minicpm-o2.6)!
*   **RLAIF-V Accepted by CVPR 2025:**  Learn about the alignment technique for MiniCPM-o, open-sourced [code](https://github.com/RLHF-V/RLAIF-V), [dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset), and [paper](https://arxiv.org/abs/2405.17220).

## Key Benefits:

*   **Powerful Multimodal Capabilities**:  Process and understand images, videos, audio, and text.
*   **Efficient Deployment**: Designed for on-device use, enabling high performance on phones, tablets, and other edge devices.
*   **Open-Source and Accessible**: Leverage the power of advanced MLLMs without proprietary limitations.
*   **Versatile Use Cases**: Image and video understanding, real-time speech conversation, multimodal live streaming, and more.

## Contents

-   [MiniCPM-V 4.5](#minicpm-v-45)
    -   [Inference Efficiency](#inference-efficiency)
-   [MiniCPM-o 2.6](#minicpm-o-26)
-   [MiniCPM-V & o Cookbook](#minicpm-v--o-cookbook)
-   [Chat with Our Demo on Gradio 🤗](#chat-with-our-demo-on-gradio-)
-   [Inference](#inference)
    -   [Model Zoo](#model-zoo)
    -   [Multi-turn Conversation](#multi-turn-conversation)
        -   [Chat with Multiple Images](#chat-with-multiple-images)
        -   [In-context Few-shot Learning](#in-context-few-shot-learning)
        -   [Chat with Video](#chat-with-video)
        -   [Speech and Audio Mode](#speech-and-audio-mode)
        -   [Multimodal Live Streaming](#multimodal-live-streaming)
    -   [Inference on Multiple GPUs](#inference-on-multiple-gpus)
    -   [Inference on Mac](#inference-on-mac)
    -   [Efficient Inference with llama.cpp, Ollama, vLLM](#efficient-inference-with-llamacpp-ollama-vllm)
-   [Fine-tuning](#fine-tuning)
-   [Awesome work using MiniCPM-V & MiniCPM-o](#awesome-work-using-minicpm-v--minicpm-o)
-   [FAQs](#faqs)
-   [Limitations](#limitations)

## MiniCPM-V 4.5

**MiniCPM-V 4.5** is the latest and most capable model in the MiniCPM-V series, demonstrating significant advancements in vision-language capabilities. Built on Qwen3-8B and SigLIP2-400M with 8B parameters, this model delivers exceptional performance.

### Key Features

*   **State-of-the-Art Vision-Language Capability:**
    *   Achieves an average score of 77.0 on OpenCompass, surpassing GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B.
*   **Efficient High-FPS and Long Video Understanding:**
    *   Utilizes a unified 3D-Resampler for a 96x compression rate of video tokens.
    *   Supports high-FPS (up to 10FPS) and long video understanding.
*   **Controllable Hybrid Fast/Deep Thinking:**
    *   Supports both fast and deep thinking modes for efficient and complex problem-solving.
*   **Strong OCR, Document Parsing, and Multilingual Support:**
    *   Based on [LLaVA-UHD](https://arxiv.org/pdf/2403.11703) architecture.
    *   Achieves leading performance on OCRBench.
    *   Supports PDF document parsing on OmniDocBench.
    *   Features trustworthy behaviors and multilingual capabilities in 30+ languages.
*   **Easy Usage:**
    *   Easy to use in various ways with [llama.cpp](https://github.com/tc-mb/llama.cpp/blob/Support-MiniCPM-V-4.5/docs/multimodal/minicpmv4.5.md) and [ollama](https://github.com/tc-mb/ollama/tree/MIniCPM-V) support for efficient CPU inference, [int4](https://huggingface.co/openbmb/MiniCPM-V-4_5-int4), [GGUF](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) and [AWQ](https://github.com/tc-mb/AutoAWQ) format quantized models, [SGLang](https://github.com/tc-mb/sglang/tree/main) and [vLLM](#efficient-inference-with-llamacpp-ollama-vllm) support for high-throughput and memory-efficient inference, fine-tuning with [Transformers](https://github.com/tc-mb/transformers/tree/main) and [LLaMA-Factory](./docs/llamafactory_train_and_infer.md), quick [local WebUI demo](#chat-with-our-demo-on-gradio), optimized [local iOS app](https://github.com/tc-mb/MiniCPM-o-demo-iOS), and online web demo on [server](http://101.126.42.235:30910/).

### Key Techniques

<div align="center">
<img src="./assets/minicpm-v-4dot5-framework.png" , width=100%>
</div>

*   **Unified 3D-Resampler for High-density Video Compression:** Enables high-FPS and long video understanding by compressing video tokens.
*   **Pre-training:** Unified learning for OCR and document knowledge.
*   **Post-training:** Hybrid Fast/Deep Thinking with multimodal RL for balanced reasoning.

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
              <td><b>73.6</td>
              <td>2.63h</td>
              <td>32G</td>
          </tr>
          <tr>
              <td nowrap="nowrap" align="left">MiniCPM-V 4.5</td>
              <td>8.7B</td>
              <td>73.5</td>
              <td><b>0.26h</td>
              <td><b>28G</td>
        </tr>
    </tbody>
</table>
</div>

Both Video-MME and OpenCompass were evaluated using 8×A100 GPUs for inference. The reported inference time of Video-MME includes full model-side computation, and excludes the external cost of video frame extraction (dependent on specific frame extraction tools) for fair comparison.

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

We deploy MiniCPM-V 4.5 on iPad M4 with [iOS demo](https://github.com/tc-mb/MiniCPM-o-demo-iOS). The demo video is the raw screen recording without edition.

<table align="center"> 
    <p align="center">
      <img src="assets/minicpmv4_5/v45_en_handwriting.gif" width=45%/>
      &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="assets/minicpmv4_5/v45_en_cot.gif" width=45%/>
    </p>
    <p align="center">
      <img src="assets/minicpmv4_5/v45_cn_handwriting.gif" width=45%/>
      &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="assets/minicpmv4_5/v45_cn_travel.gif" width=45%/>
    </p>
</table>

## MiniCPM-o 2.6

**MiniCPM-o 2.6** is the latest end-to-end multimodal model in the MiniCPM series, designed to achieve GPT-4o-level performance in vision, speech, and live streaming scenarios.

### Key Features

*   **Leading Visual Capability:**
    *   Achieves an average score of 70.2 on OpenCompass.
    *   Outperforms GPT-4o-202405, Gemini 1.5 Pro, and Claude 3.5 Sonnet in single-image understanding.
    *   Outperforms GPT-4V and Claude 3.5 Sonnet in multi-image and video understanding.
*   **State-of-the-art Speech Capability:**
    *   Supports bilingual real-time speech conversation with configurable voices.
    *   Outperforms GPT-4o-realtime in ASR and STT translation.
    *   Delivers state-of-the-art performance in speech conversation.
    *   Supports voice cloning and creative voice generation.
*   **Strong Multimodal Live Streaming Capability:**
    *   Processes continuous video and audio streams for real-time interaction.
    *   Outperforms GPT-4o-202408 and Claude 3.5 Sonnet on StreamingBench.
*   **Strong OCR Capability and Others:**
    *   Processes images with any aspect ratio.
    *   Achieves state-of-the-art OCRBench performance.
    *   Features trustworthy behaviors and multilingual support.
*   **Superior Efficiency:**
    *   High token density improves inference speed, memory usage, and power consumption.
    *   Supports multimodal live streaming on end-side devices.
*   **Easy Usage:**
    *   Can be easily used in various ways with [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md) support for efficient CPU inference, [int4](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) and [GGUF](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf) format quantized models, [vLLM](#efficient-inference-with-llamacpp-ollama-vllm) support, fine-tuning with [LLaMA-Factory](./docs/llamafactory_train_and_infer.md), quick [local WebUI demo](#chat-with-our-demo-on-gradio), and online web demo on [server](https://minicpm-omni-webdemo-us.modelbest.cn/).

**Model Architecture.**

-   **End-to-end Omni-modal Architecture.** Different modality encoders/decoders are connected and trained in an **end-to-end** fashion to fully exploit rich multimodal knowledge. The model is trained in a fully end-to-end manner with only CE loss.
-   **Omni-modal Live Streaming Mechanism.** (1) We change the offline modality encoder/decoders into online ones for **streaming inputs/outputs.** (2) We devise a **time-division multiplexing (TDM) mechanism** for omni-modality streaming processing in the LLM backbone. It divides parallel omni-modality streams into sequential info within small periodic time slices.
-   **Configurable Speech Modeling Design.** We devise a multimodal system prompt, including traditional text system prompt, and **a new audio system prompt to determine the assistant voice**. This enables flexible voice configurations in inference time, and also facilitates end-to-end voice cloning and description-based voice creation.

<div align="center">
<img src="./assets/minicpm-o-26-framework-v2.png" , width=80%>
</div>

### Evaluation

<div align="center">
  <img src="./assets/radar.jpg", width=80%>
</div>

<details>
<summary>Click to view visual understanding results.</summary>

**Image Understanding**

<div align="center">
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>Token Density<sup>+</sup></th>
            <th>OpenCompass</th>
            <th>OCRBench</th>
            <th>MathVista mini</th>
            <th>ChartQA</th>
            <th>MMVet</th>
            <th>MMStar</th>
            <th>MME</th>
            <th>MMB1.1 test</th>
            <th>AI2D</th>
            <th>MMMU val</th>
            <th>HallusionBench</th>
            <th>TextVQA val</th>
            <th>DocVQA test</th>
            <th>MathVerse mini</th>
            <th>MathVision</th>
            <th>MMHal Score</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="19" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o-20240513</td>
            <td>-</td>
            <td>1088</td>
            <td><u>69.9</u></td>
            <td>736</td>
            <td>61.3</td>
            <td>85.7</td>
            <td><strong>69.1</strong></td>
            <td>63.9</td>
            <td>2328.7</td>
            <td>82.2</td>
            <td>84.6</td>
            <td><strong>69.2</strong></td>
            <td><strong>55.0</strong></td>
            <td>-</td>
            <td>92.8</td>
            <td><strong>50.2</strong></td>
            <td><strong>30.4</strong></td>
            <td><u>3.6</u></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Claude3.5-Sonnet</td>
            <td>-</td>
            <td>750</td>
            <td>67.9</td>
            <td>788</td>
            <td>61.6</td>
            <td><strong>90.8</strong></td>
            <td>66.0</td>
            <td>62.2</td>
            <td>1920.0</td>
            <td>78.5</td>
            <td>80.2</td>
            <td><u>65.9</u></td>
            <td>49.9</td>
            <td>-</td>
            <td><strong>95.2</strong></td>
            <td>-</td>
            <td>-</td>
            <td>3.4</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Gemini 1.5 Pro</td>
            <td>-</td>
            <td>-</td>
            <td>64.4</td>
            <td>754</td>
            <td>57.7</td>
            <td>81.3</td>
            <td>64.0</td>
            <td>59.1</td>
            <td>2110.6</td>
            <td>73.9</td>
            <td>79.1</td>
            <td>60.6</td>
            <td>45.6</td>
            <td>73.5</td>
            <td>86.5</td>
            <td>-</td>
            <td>19.2</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o-mini-20240718</td>
            <td>-</td>
            <td>1088</td>
            <td>64.1</td>
            <td>785</td>
            <td>52.4</td>
            <td>-</td>
            <td>66.9</td>
            <td>54.8</td>
            <td>2003.4</td>
            <td>76.0</td>
            <td>77.8</td>
            <td>60.0</td>
            <td>46.1</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>3.3</td>
        </tr>
        <tr>
            <td colspan="19" align="left"><strong>Open Source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Cambrian-34B</td>
            <td>34B</td>
            <td><u>1820</u></td>
            <td>58.3</td>
            <td>591</td>
            <td>50.3</td>
            <td>75.6</td>
            <td>53.2</td>
            <td>54.2</td>
            <td>2049.9</td>
            <td>77.8</td>
            <td>79.5</td>
            <td>50.4</td>
            <td>41.6</td>
            <td>76.7</td>
            <td>75.5</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GLM-4V-9B</td>
            <td>13B</td>
            <td>784</td>
            <td>59.1</td>
            <td>776</td>
            <td>51.1</td>
            <td>-</td>
            <td>58.0</td>
            <td>54.8</td>
            <td>2018.8</td>
            <td>67.9</td>
            <td>71.2</td>
            <td>46.9</td>
            <td>45.0</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Pixtral-12B</td>
            <td>12B</td>
            <td>256</td>
            <td>61.0</td>
            <td>685</td>
            <td>56.9</td>
            <td>81.8</td>
            <td>58.5</td>
            <td>54.5</td>
            <td>-</td>
            <td>72.7</td>
            <td>79.0</td>
            <td>51.1</td>
            <td>47.0</td>
            <td>75.7</td>
            <td>90.7</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">VITA-1.5</td>
            <td>8B</td>
            <td>784</td>
            <td>63.3</td>
            <td>741</td>
            <td>66.2</td>
            <td>-</td>
            <td>52.7</td>
            <td>60.2</td>
            <td>2328.1</td>
            <td>76.8</td>
            <td>79.2</td>
            <td>52.6</td>
            <td>44.6</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">DeepSeek-VL2-27B (4B)</td>
            <td>27B</td>
            <td>672</td>
            <td>66.4</td>
            <td>809</td>
            <td>63.9</td>
            <td>86.0</td>
            <td>60.0</td>
            <td>61.9</td>
            <td>2253.0</td>
            <td>81.2</td>
            <td>83.8</td>
            <td>54.0</td>
            <td>45.3</td>
            <td><u>84.2</u></td>
            <td>93.3</td>
            <td>-</td>
            <td>-</td>
            <td>3.0</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen2-VL-7B</td>
            <td>8B</td>
            <td>784</td>
            <td>67.1</td>
            <td><u>866</u></td>
            <td>58.2</td>
            <td>83.0</td>
            <td>62.0</td>
            <td>60.7</td>
            <td>2326.0</td>
            <td>81.8</td>
            <td>83.0</td>
            <td>54.1</td>
            <td>50.6</td>
            <td><strong>84.3</strong></td>
            <td><u>94.5</u></td>
            <td>31.9</td>
            <td>16.3</td>
            <td>3.2</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-OneVision-72B</td>
            <td>72B</td>
            <td>182</td>
            <td>68.1</td>
            <td>741</td>
            <td>67.5</td>
            <td>83.7</td>
            <td>60.6</td>
            <td><strong>65.8</strong></td>
            <td>2261.0</td>
            <td><strong>85.0</strong></td>
            <td><u>85.6</u></td>
            <td>56.8</td>
            <td>49.0</td>
            <td>80.5</td>
            <td>91.3</td>
            <td>39.1</td>
            <td>-</td>
            <td>3.5</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2.5-8B</td>
            <td>8B</td>
            <td>706</td>
            <td>68.3</td>
            <td>822</td>
            <td><u>64.4</u></td>
            <td>84.8</td>
            <td>62.8</td>
            <td>62.8</td>
            <td>2344.0</td>
            <td><u>83.6</u></td>
            <td>84.5</td>
            <td>56.0</td>
            <td>50.1</td>
            <td>79.1</td>
            <td>93.0</td>
            <td>39.5</td>
            <td>19.7</td>
            <td>3.4</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td><strong>2822</strong></td>
            <td>65.2</td>
            <td>852*</td>
            <td>60.6</td>
            <td>79.4</td>
            <td>60.0</td>
            <td>57.5</td>
            <td><u>2348.4*</u></td>
            <td>78.0</td>
            <td>82.1</td>
            <td>49.8*</td>
            <td>48.1*</td>
            <td>80.1</td>
            <td>90.8</td>
            <td>25.7</td>
            <td>18.3</td>
            <td>3.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiniCPM-o 2.6</td>
            <td>8B</td>
            <td><strong>2822</strong></td>
            <td><strong>70.2</strong></td>
            <td><strong>897*</strong></td>
            <td><strong>71.9*</strong></td>
            <td><u>86.9*</u></td>
            <td><u>67.5</u></td>
            <td><u>64.0</u></td>
            <td><strong>2372.0*</strong></td>
            <td>80.5</td>
            <td><strong>85.8</strong></td>
            <td>50.4*</td>
            <td><u>51.9</u></td>
            <td>82.0</td>
            <td>93.5</td>
            <td><u>41.4*</u></td>
            <td><u>23.1*</u></td>
            <td><strong>3.8</strong></td>
        </tr>
    </tbody>
</table>
</div>
* We evaluate this benchmark using chain-of-thought prompting. Specifically, for MME, we used this technique only for the Cognition set.


<sup>+</sup> Token Density: number of pixels encoded into each visual token at maximum resolution, i.e., # pixels at maximum resolution / # visual tokens.

Note: For proprietary models, we calculate token density based on the image encoding charging strategy defined in the official API documentation, which provides an upper-bound estimation.


**Multi-image and Video Understanding**

<div align="center">
 
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>BLINK val</th>
            <th>Mantis Eval</th>
            <th>MIRB</th>
            <th>Video-MME (wo / w subs)</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="6" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o-20240513</td>
            <td>-</td>
            <td><strong>68.0</strong></td>
            <td>-</td>
            <td>-</td>
            <td><strong>71.9/77.2<strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT4V</td>
            <td>-</td>
            <td>54.6</td>
            <td>62.7</td>
            <td>53.1</td>
            <td>59.9/63.3</td>
        </tr>
        <tr>
            <td colspan="6" align="left"><strong>Open-source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">VITA-1.5</td>
            <td>8B</td>
            <td>45.0</td>
            <td>-</td>
            <td>-</td>
            <td>56.1/58.7</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-Interleave 14B</td>
            <td>14B</td>
            <td>52.6</td>
            <td>66.4</td>
            <td>30.2</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-OneVision-72B</td>
            <td>72B</td>
            <td>55.4</td>
            <td><strong>77.6</strong></td>
            <td>-</td>
            <td><u>66.2/69.5</u></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MANTIS 8B</td>
            <td>8B</td>
            <td>49.1</td>
            <td>59.5</td>
            <td>34.8</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen2-VL-7B</td>
            <td>8B</td>
            <td>53.2</td>
            <td>69.6*</td>
            <td><strong>67.6*</strong></td>
            <td>63.3/69.0</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2.5-8B</td>
            <td>8B</td>
            <td>54.8</td>
            <td>67.7</td>
            <td>52.5</td>
            <td>64.2/66.9</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td>53.0</td>
            <td>69.1</td>
            <td>53.8</td>
            <td>60.9/63.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiniCPM-o 2.6</td>
            <td>8B</td>
            <td><u>56.7</u></td>
            <td><u>71.9</u></td>
            <td><u>58.6</u></td>
            <td>63.9/67.9</td>
        </tr>
    </tbody>
</table>

</div>
* We evaluate officially released checkpoints by ourselves.

</details>

<details>
<summary>Click to view audio understanding and speech conversation results.</summary>

**Audio Understanding**

<div align="center">
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Task</th>
            <th>Size</th>
            <th colspan="3">ASR (zh)</th>
            <th colspan="3">ASR (en)</th>
            <th colspan="2">AST</th>
            <th>Emotion</th>
        </tr>
        <tr>
            <th align="left">Metric</th>
            <td></td>
            <th colspan="3">CER↓</th>
            <th colspan="3">WER↓</th>
            <th colspan="2">BLEU↑</th>
            <th>ACC↑</th>
        </tr>
        <tr>
            <th align="left">Dataset</th>
            <td></td>
            <th>AISHELL-1</th>
            <th>Fleurs zh</th>
            <th>WenetSpeech test-net</th>
            <th>LibriSpeech test-clean</th>
            <th>GigaSpeech</th>
            <th>TED-LIUM</th>
            <th>CoVoST en2zh</th>
            <th>CoVoST zh2en</th>
            <th>MELD emotion</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="11" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o-Realtime</td>
            <td>-</td>
            <td>7.3*</td>
            <td><u>5.4*</u></td>
            <td>28.9*</td>
            <td>2.6*</td>
            <td>12.9*</td>
            <td>4.8*</td>
            <td>37.1*</td>
            <td>15.7*</td>
            <td>33.2*</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Gemini 1.5 Pro</td>
            <td>-</td>
            <td>4.5*</td>
            <td>5.9*</td>
            <td>14.3*</td>
            <td>2.9*</td>
            <td>10.6*</td>
            <td><strong>3.0*</strong></td>
            <td><u>47.3*</u></td>
            <td>22.6*</td>
            <td>48.4*</td>
        </tr>
        <tr>
            <td colspan="11" align="left"><strong>Open-Source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen2-Audio-