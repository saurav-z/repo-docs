<div align="center">
  <img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" alt="MiniCPM-V and MiniCPM-o Title" />
</div>

# MiniCPM: Experience GPT-4o Level Multimodal Understanding on Your Device

**[Open Source Repo](https://github.com/OpenBMB/MiniCPM-V)** | **[中文](./README_zh.md)** | **English**

> Unleash the power of multimodal understanding with MiniCPM, a series of efficient and performant multimodal large language models (MLLMs) designed for on-device deployment.

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

## Key Features

*   🚀 **State-of-the-Art Performance:** Achieves GPT-4o level performance on single image, multi-image, video and audio understanding tasks.
*   📱 **On-Device Deployment:** Designed for efficient deployment on mobile phones and other edge devices.
*   🖼️ **Multimodal Input:** Supports images, videos, audio, and text inputs.
*   🗣️ **High-Quality Outputs:** Generates high-quality text and speech outputs.
*   🌐 **Multilingual Support:** Excellent performance in over 30 languages.
*   🛠️ **Easy to Use:** Supports various inference methods, quantization, and fine-tuning.
*   💡 **Open Source:**  Freely available for academic and commercial use.

## Table of Contents

*   [MiniCPM-V 4.5](#minicpm-v-45)
    *   [Key Techniques](#key-techniques)
    *   [Inference Efficiency](#inference-efficiency)
    *   [Examples](#examples)
*   [MiniCPM-o 2.6](#minicpm-o-26)
    *   [Key Techniques](#key-techniques)
    *   [Examples](#examples)
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
    *   [Simple Fine-tuning](#simple-fine-tuning)
    *   [With Align-Anything](#with-align-anything)
    *   [With LLaMA-Factory](#with-llama-factory)
    *   [With the SWIFT Framework](#with-the-swift-framework)
*   [Awesome Work using MiniCPM-V & MiniCPM-o](#awesome-work-using-minicpm-v--minicpm-o)
*   [FAQs](#faqs)
*   [Limitations](#limitations)
*   [Model License](#model-license)
*   [Statement](#statement)
*   [Institutions](#institutions)
*   [🌟 Star History](#-star-history)
*   [Key Techniques and Other Multimodal Projects](#key-techniques-and-other-multimodal-projects)
*   [Citation](#citation)

---

## MiniCPM-V 4.5

MiniCPM-V 4.5, the latest in the MiniCPM-V series, is an 8B parameter model built on Qwen3-8B and SigLIP2-400M. It's designed to excel in various multimodal tasks, outperforming GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language capabilities.

*   **Key Improvements:**
    *   **State-of-the-art Vision-Language Capability:** Outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B.
    *   **Efficient Video Understanding:** Features high-FPS and long video understanding with up to a 96x compression rate for video tokens.
    *   **Controllable Hybrid Thinking:** Supports both fast and deep thinking modes.
    *   **Strong OCR, Document Parsing & Multilingual Capabilities.**
    *   **Easy Usage:** Supports [llama.cpp](https://github.com/tc-mb/llama.cpp/blob/Support-MiniCPM-V-4.5/docs/multimodal/minicpmv4.5.md), [Ollama](https://github.com/tc-mb/ollama/tree/MIniCPM-V), int4, GGUF and AWQ format quantized models, [SGLang](https://github.com/tc-mb/sglang/tree/main) and [vLLM](#efficient-inference-with-llamacpp-ollama-vllm) support for high-throughput and memory-efficient inference, fine-tuning on new domains and tasks with [Transformers](https://github.com/tc-mb/transformers/tree/main) and [LLaMA-Factory](./docs/llamafactory_train_and_infer.md), quick [local WebUI demo](#chat-with-our-demo-on-gradio), optimized [local iOS app](https://github.com/tc-mb/MiniCPM-o-demo-iOS) on iPhone and iPad, and online web demo on [server](http://101.126.42.235:30910/).

### Key Techniques

*   **Unified 3D-Resampler for High-density Video Compression:** This 3D-Resampler allows MiniCPM-V 4.5 to process more video frames efficiently. It groups and compresses up to 6 video frames into 64 tokens, achieving a 96x compression rate. This supports high-FPS video understanding.
*   **Unified Learning for OCR and Document Knowledge:** The model learns to switch between accurate text recognition and multimodal knowledge reasoning.
*   **Hybrid Fast/Deep Thinking with Multimodal RL:** Provides a balanced reasoning experience through switchable fast and deep thinking modes.

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
            <td><b>77.0</b>
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

---

## MiniCPM-o 2.6

MiniCPM-o 2.6 is the latest in the MiniCPM-o series, an 8B parameter end-to-end model built on SigLip-400M, Whisper-medium-300M, ChatTTS-200M, and Qwen2.5-7B. It provides a level of performance comparable to GPT-4o for vision, speech, and multimodal live streaming on end-side devices.

*   **Key Features:**
    *   **Leading Visual Capability:** Achieves an average score of 70.2 on OpenCompass, surpassing GPT-4o-202405, Gemini 1.5 Pro, and Claude 3.5 Sonnet.
    *   **State-of-the-art Speech Capability:** Supports bilingual real-time speech conversation with configurable voices. It outperforms GPT-4o-realtime on audio understanding tasks and shows state-of-the-art performance on speech conversation.
    *   **Strong Multimodal Live Streaming:** Supports real-time speech interaction with continuous video and audio streams. It shows state-of-the-art performance on StreamingBench.
    *   **Superior Efficiency:** Offers state-of-the-art token density, improving inference speed, latency, and memory usage.
    *   **Easy Usage:**  Supports [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md), int4, GGUF, [vLLM](#efficient-inference-with-llamacpp-ollama-vllm), [LLaMA-Factory](./docs/llamafactory_train_and_infer.md), and provides a [local WebUI demo](#chat-with-our-demo-on-gradio), and online web demo on [server](https://minicpm-omni-webdemo-us.modelbest.cn/).

### Key Techniques

*   **End-to-end Omni-modal Architecture:** Connects different modality encoders/decoders in an end-to-end fashion.
*   **Omni-modal Live Streaming Mechanism:** Employs a time-division multiplexing (TDM) mechanism for real-time streaming processing in the LLM backbone.
*   **Configurable Speech Modeling Design:** Includes a multimodal system prompt for flexible voice configurations.

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

---

## MiniCPM-V & o Cookbook

The MiniCPM-V & o Cookbook provides comprehensive, ready-to-deploy solutions for the MiniCPM-V and MiniCPM-o model series. It allows developers to quickly implement multimodal AI applications with vision, speech, and live-streaming capabilities.

*   **Key Features:**
    *   **Easy Usage Documentation:** Provides clear and organized documentation.
    *   **Broad User Spectrum:** Supports a wide range of users, from individuals to enterprises and researchers.
    *   **Versatile Deployment Scenarios:** Offers optimal solutions for various hardware environments and deployment needs.

    *   **Individuals**: Enjoy effortless inference using [Ollama](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_ollama.md) and [Llama.cpp](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_llamacpp.md) with minimal setup.
    *   **Enterprises**: Achieve high-throughput, scalable performance with [vLLM](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_vllm.md) and [SGLang](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/MiniCPM-v4_sglang.md).
    *   **Researchers**: Leverage advanced frameworks including [Transformers](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/finetune_full.md), [LLaMA-Factory](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/finetune_llamafactory.md), [SWIFT](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/swift.md), and [Align-anything](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/align_anything.md) to enable flexible model development and cutting-edge experimentation.
    *   **Web demo**: Launch interactive multimodal AI web demo with [FastAPI](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/demo/README.md).
    *   **Quantized deployment**: Maximize efficiency and minimize resource consumption using [GGUF](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/gguf/minicpm-v4_gguf_quantize.md) and [BNB](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/bnb/minicpm-v4_bnb_quantize.md).
    *   **End devices**: Bring powerful AI experiences to [iPhone and iPad](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/demo/ios_demo/ios.md), supporting offline and privacy-sensitive applications.

---

## Chat with Our Demo on Gradio 🤗

Experiment with MiniCPM through our online demos, built using Hugging Face Gradio.

*   [MiniCPM-o 2.6](https://minicpm-omni-webdemo-us.modelbest.cn/) | [MiniCPM-V 2.6](http://120.92.209.146:8887/) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2).

### Local WebUI Demo

Quickly create your own local WebUI demo using the following commands.

Please ensure that `transformers==4.44.2` is installed, as other versions may have compatibility issues.

If you are using an older version of PyTorch, you might encounter this issue `"weight_norm_fwd_first_dim_kernel" not implemented for 'BFloat16'`, Please add `self.minicpmo_model.tts.float()` during the model initialization.

**For real-time voice/video call demo:**

1.  launch model server:

```shell
pip install -r requirements_o2.6.txt

python web_demos/minicpm-o_2.6/model_server.py
```

2.  launch web server:

```shell
# Make sure Node and PNPM is installed.
sudo apt-get update
sudo apt-get install nodejs npm
npm install -g pnpm


cd web_demos/minicpm-o_2.6/web_server
# create ssl cert for https, https is required to request camera and microphone permissions.
bash ./make_ssl_cert.sh  # output key.pem and cert.pem

pnpm install  # install requirements
pnpm run dev  # start server
```

Open `https://localhost:8088/` in browser and enjoy the real-time voice/video call.

**For chatbot demo:**

```shell
pip install -r requirements_o2.6.txt

python web_demos/minicpm-o_2.6/chatbot_web_demo_o2.6.py
```

Open `http://localhost:8000/` in browser and enjoy the vision mode chatbot.

---

## Inference

### Model Zoo

| Model           | Device | Memory    | Description                                                                                                                                  | Download                                                                                                                                                           |
| :-------------- | :----- | :-------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MiniCPM-V 4.5   | GPU    | 18 GB     | The latest version, strong end-side multimodal performance for single image, multi-image and video understanding.                           | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5) |
| MiniCPM-V 4.5 gguf | CPU  | 8 GB      | The gguf version, lower memory usage and faster inference.                                                                                | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-gguf) |
| MiniCPM-V 4.5 int4 | GPU  | 9 GB      | The int4 quantized version, lower GPU memory usage.                                                                                         | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4) |
| MiniCPM-V 4.5 AWQ  | GPU   | 9 GB      | The int4 quantized version, lower GPU memory usage.                                                                                         | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-AWQ) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-AWQ) |
| MiniCPM-o 2.6   | GPU    | 18 GB     | The latest version, achieving GPT-4o level performance for vision, speech and multimodal live streaming on end-side devices.                  | [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6) |
| MiniCPM-o 2.6 gguf | CPU   | 8 GB      | The gguf version, lower memory usage and faster inference.                                                                                | [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-gguf) |
| MiniCPM-o 2.6 int4 | GPU   | 9 GB      | The int4 quantized version, lower GPU memory usage.                                                                                         | [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-int4) |

### Multi-turn Conversation

If you wish to enable long-thinking mode, provide the argument `enable_thinking=True` to the chat function.

```shell
pip install -r requirements_o2.6.txt
```

Please refer to the following codes to run.

<div align="center">
<img src="assets/minicpmo2_6/show_demo.jpg" width="500px">
</div>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(100)

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

image = Image.open('./assets/minicpmo2_6/show_demo.jpg').convert('RGB')

enable_thinking=False # If `enable_thinking=True`, the long-thinking mode is enabled.

# First round chat 
question = "What is the landform in the picture?"
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    enable_thinking=enable_thinking
)
print(answer)

# Second round chat, pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["What should I pay attention to when traveling here?"]})

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```

You will get the following output:

```shell
# round1
The landform in the picture is karst topography. Karst landscapes are characterized by distinctive, jagged limestone hills or mountains with steep, irregular peaks and deep valleys—exactly what you see here These unique formations result from the dissolution of soluble rocks like limestone over millions of years through water erosion.

This scene closely resembles the famous karst landscape of Guilin and Yangshuo in China’s Guangxi Province. The area features dramatic, pointed limestone peaks rising dramatically above serene rivers and lush green forests, creating a breathtaking and iconic natural beauty that attracts millions of visitors each year for its picturesque views.

# round2
When traveling to a karst landscape like this, here are some important tips:

1. Wear comfortable shoes: The terrain can be uneven and hilly.
2. Bring water and snacks for energy during hikes or boat rides.
3. Protect yourself from the sun with sunscreen, hats, and sunglasses—especially since you’ll likely spend time outdoors exploring scenic spots.
4. Respect local customs and nature regulations by not littering or disturbing wildlife.

By following these guidelines, you'll have a safe and enjoyable trip while appreciating the stunning natural beauty of places such as Guilin’s karst mountains.
```

#### Chat with Multiple Images
<details>
<summary> Click to view Python code running MiniCPM-V-4_5 with multiple images input. </summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True,  # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True)  # or openbmb/MiniCPM-o-2_6

image1 = Image.open('image1.jpg').convert('RGB')
image2 = Image.open('image2.jpg').convert('RGB')
question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'

msgs = [{'role': 'user', 'content': [image1, image2, question]}]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```
</details>

#### In-context Few-shot Learning
<details>
<summary> Click to view Python code running MiniCPM-V-4_5 with few-shot input. </summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True,  # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True)  # or openbmb/MiniCPM-o-2_6

question = "production date" 
image1 = Image.open('example1.jpg').convert('RGB')
answer1 = "2023.08.04"
image2 = Image.open('example2.jpg').convert('RGB')
answer2 = "2007.04.24"
image_test = Image.open('test.jpg').convert('RGB')

msgs = [
    {'role': 'user', 'content': [image1, question]}, {'role': 'assistant', 'content': [answer1]},
    {'role': 'user', 'content': [image2, question]}, {'role': 'assistant', 'content': [answer2]},
    {'role': 'user', 'content': [image_test, question]}
]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```
</details>

#### Chat with Video
<details>
<summary> Click to view Python code running MiniCPM-V-4_5 by with video input and 3D-Resampler. </summary>

```python
## The 3d-resampler compresses multiple frames into 64 tokens by introducing temporal_ids. 
# To achieve this, you need to organize your video data into two corresponding sequences: 
#   frames: List[Image]
#   temporal_ids: List[List[Int]].

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
from scipy.spatial import cKDTree
import numpy as np
import math

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True,  # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True)  # or openbmb/MiniCPM-o-2_6

MAX_NUM_FRAMES=180 # Indicates the maximum number of frames received after the videos are packed. The actual maximum number of valid frames is MAX_NUM_FRAMES * MAX_NUM_PACKING.
MAX_NUM_PACKING=3  # indicates the maximum packing number of video frames. valid range: 1-6
TIME_SCALE = 0.1 

def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def encode_video(video_path, choose_fps=3, force_packing=None):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps
        
    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
        
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = [i for i in range(0, len(vr))]      
    frame_idx =  np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)
    
    print(video_path, ' duration:', video_duration)
    print(f'get video frames={len(frame_idx)}, packing_nums={packing_nums}')
    
    frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame