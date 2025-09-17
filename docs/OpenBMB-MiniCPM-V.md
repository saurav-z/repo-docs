<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img> 

**MiniCPM-V: Unleashing GPT-4o Level Multimodal Understanding on Your Device**

  <strong>[中文](./README_zh.md) |
  English</strong>

</div>

<p align="center">
   <a href="https://github.com/OpenBMB/MiniCPM-V"> 🔗 **View the Original Repository** </a>
</p>

## **Introduction**

MiniCPM-V is a series of efficient, end-side multimodal LLMs (MLLMs) designed to bring advanced image, video, and text understanding to your phone or device. Built upon cutting-edge research, MiniCPM-V models deliver high-quality text outputs, with the latest version achieving **GPT-4o level performance**. MiniCPM-o expands this capability by also incorporating audio inputs for high-quality speech outputs.

## **Key Features**

*   **🔥 Superior Performance**: MiniCPM-V models excel in vision-language tasks, outperforming leading open-source and even proprietary models, with MiniCPM-V 4.5 exceeding GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B on vision-language capabilities.
*   **🎬 Advanced Video Understanding**: Experience efficient high-FPS and long video understanding capabilities through advanced techniques like unified 3D-Resampling, enabling the models to process significantly more video frames with reduced LLM computational cost.
*   **⚙️ Controllable Reasoning**: Switch between efficient "fast thinking" and in-depth "deep thinking" modes to optimize for different user scenarios and task complexities.
*   **💪 Strong OCR and Document Processing**: Exceptional performance in OCR (Optical Character Recognition) and document parsing, with MiniCPM-V 4.5 achieving leading performance on OCRBench.
*   **🎙️ Enhanced Audio and Speech Capabilities**: MiniCPM-o 2.6 supports bilingual real-time speech conversation with configurable voices, voice cloning, and multimodal live streaming capabilities, delivering superior performance to previous iterations.
*   **🚀 Optimized Deployment**: The models are designed for efficient deployment on various devices, including phones, iPads, and Macs, and are supported by popular frameworks such as llama.cpp, vLLM, Ollama, and SGLang.

## **Recent Updates**

*   **MiniCPM-V 4.5 Open-Sourced**: The latest version, offering state-of-the-art multimodal performance and new features, is now available.
*   **MiniCPM-o 2.6 Open-Sourced**:  Achieves GPT-4o-level performance for vision, speech and multimodal live streaming on end-side devices.
*   **Cookbook & Docs**:  Comprehensive guides and a documentation website are available to help you get started.
*   **Framework Support**:  Full support for popular frameworks such as [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15575), [vLLM](https://github.com/vllm-project/vllm/pull/23586), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/pull/9022).

## **Contents**

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

## **MiniCPM-V 4.5**

MiniCPM-V 4.5 is the latest model in the MiniCPM-V series and is built on the Qwen3-8B architecture. With 8B parameters, it is optimized for efficient and effective processing of images, videos and text.

### **Key Techniques**

*   **Unified 3D-Resampler:** Enables high-density video compression, allowing the model to process more video frames.
*   **Unified Learning:** Improves OCR capabilities and knowledge extraction from documents.
*   **Hybrid Fast/Deep Thinking:** Balances reasoning efficiency and problem-solving depth.

### **Inference Efficiency**

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


### **Examples**

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


## **MiniCPM-o 2.6**

MiniCPM-o 2.6 is a model built in an end-to-end fashion, offering leading performance in visual, speech, and multimodal live streaming. It achieves GPT-4o level performance.

### Evaluation
<div align="center">
  <img src="./assets/radar.jpg", width=80%>
</div>

### Examples <!-- omit in toc -->

We deploy MiniCPM-o 2.6 on end devices. The demo video is the raw-speed recording on an iPad Pro and a Web demo.

<div align="center">
  <a href="https://www.youtube.com/watch?v=vRIMbxJzStY&t=2s"><img src="./assets/minicpmo2_6/2dot6_o_demo_video_img.png", width=70%></a>
</div>

<br>

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmo2_6/minicpmo2_6_math_intersect.png" alt="math" style="margin-bottom: 5px;">
  <img src="assets/minicpmo2_6/minicpmo2_6_diagram_train_NN.png" alt="diagram" style="margin-bottom: 5px;">
  <img src="assets/minicpmo2_6/minicpmo2_6_multi-image_bike.png" alt="bike" style="margin-bottom: 5px;">
</div>


## **MiniCPM-V & o Cookbook**

Explore comprehensive, ready-to-deploy solutions for the MiniCPM-V and MiniCPM-o models in our structured [cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook), which empowers developers to rapidly implement multimodal AI applications with integrated vision, speech, and live-streaming capabilities.

## **Chat with Our Demo on Gradio 🤗**

Try our online demos powered by Hugging Face Gradio.

### Online Demo <!-- omit in toc --> 

Click here to try out the online demo of [MiniCPM-o 2.6](https://minicpm-omni-webdemo-us.modelbest.cn/) | [MiniCPM-V 2.6](http://120.92.209.146:8887/) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2).

### Local WebUI Demo <!-- omit in toc --> 
  
You can easily build your own local WebUI demo using the following commands.

Please ensure that `transformers==4.44.2` is installed, as other versions may have compatibility issues.

If you are using an older version of PyTorch, you might encounter this issue `"weight_norm_fwd_first_dim_kernel" not implemented for 'BFloat16'`, Please add `self.minicpmo_model.tts.float()` during the model initialization.

**For real-time voice/video call demo:**
1. launch model server:
```shell
pip install -r requirements_o2.6.txt

python web_demos/minicpm-o_2.6/model_server.py
```

2. launch web server:

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


## **Inference**

### **Model Zoo**

| Model           | Device | Memory    | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Description       | Download |
|:-----------|:--:|:-----------:|:-------------------|:---------------:|
| MiniCPM-V 4.5| GPU | 18 GB  | The latest version, strong end-side multimodal performance for single image, multi-image and video understanding.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5) |
| MiniCPM-V 4.5 gguf | CPU | 8 GB  | The gguf version, lower memory usage and faster inference.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-gguf) |
| MiniCPM-V 4.5 int4 | GPU | 9 GB  | The int4 quantized version, lower GPU memory usage.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4) |
| MiniCPM-V 4.5 AWQ | GPU | 9 GB  | The int4 quantized version, lower GPU memory usage.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4_5-AWQ) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-AWQ) |
| MiniCPM-o 2.6| GPU | 18 GB  | The latest version, achieving GPT-4o level performance for vision, speech and multimodal live streaming on end-side devices.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6) |
| MiniCPM-o 2.6 gguf | CPU | 8 GB  | The gguf version, lower memory usage and faster inference.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf) |
| MiniCPM-o 2.6 int4 | GPU | 9 GB  | The int4 quantized version, lower GPU memory usage.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-int4) |

### **Multi-turn Conversation**

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

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)
    
    return frames, frame_ts_id_group


video_path="video_test.mp4"
fps = 5 # fps for video
force_packing = None # You can set force_packing to ensure that 3D packing is forcibly enabled; otherwise, encode_video will dynamically set the packing quantity based on the duration.
frames, frame_ts_id_group = encode_video(video_path, fps, force_packing=force_packing)

question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]}, 
]


answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    use_image_id=False,
    max_slice_nums=1,
    temporal_ids=frame_ts_id_group
)
print(answer)
```
</details>


#### Speech and Audio Mode

Model initialization

```python
import torch
import librosa
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

model.init_tts()
model.tts.float()
```

<hr/>

##### Mimick <!-- omit in toc --> 

`Mimick` task reflects a model's end-to-end speech modeling capability. The model takes audio input, and outputs an ASR transcription and subsequently reconstructs the original audio with high similarity. The higher the similarity between the reconstructed audio and the original audio, the stronger the model's foundational capability in end-to-end speech modeling.

```python
mimick_prompt = "Please repeat each user's speech, including voice style and speech content."
audio_input, _ = librosa.load('./assets/input_examples/Trump_WEF_2018_10s.mp3', sr=16000, mono=True) # load the audio to be mimicked

# `./assets/input_examples/fast-pace.wav`, 
# `./assets/input_examples/chi-english-1.wav` 
# `./assets/input_examples/exciting-emotion.wav` 
# for different aspects of speech-centric features.

msgs = [{'role': 'user', 'content': [mimick_prompt, audio_input]}]
res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    temperature=0.3,
    generate_audio=True,
    output_audio_path='output_mimick.wav', # save the tts result to output_audio_path
)
```

<hr/>

##### General Speech Conversation with Configurable Voices <!-- omit in toc --> 

A general usage scenario of `MiniCPM-o-2.6` is role-playing a specific character based on the audio prompt. It will mimic the voice of the character to some extent and act like the character in text, including language style. In this mode, `MiniCPM-o-2.6` sounds **more natural and human-like**. Self-defined audio prompts can be used to customize the voice of the character in an end-to-end manner.


```python
ref_audio, _ = librosa.load('./assets/input_examples/icl_20.wav', sr=16000, mono=True) # load the reference audio
sys_prompt = model.get_sys_prompt(ref_audio=ref_audio, mode='audio_roleplay', language='en')

# round one
user_question = {'role': 'user', 'content': [librosa.load('xxx.wav', sr=16000, mono=True)[0]]}
msgs = [sys_prompt, user_question]
res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result_roleplay_round_1.wav',
)

# round two
history = msgs.append({'role': 'assistant', 'content': res})
user_question = {'role': 'user', 'content': [librosa.load('xxx.wav', sr=16000, mono=True)[0]]}
msgs = history.append(user_question)
res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result_roleplay_round_2.wav',
)
print(res)
```

<hr/>

##### Speech Conversation as an AI Assistant <!-- omit in toc --> 

An enhanced feature of `MiniCPM-o-2.6` is to act as an AI assistant, but only with limited choice of voices. In this mode, `MiniCPM-o-2.6` is **less human-like and more like a voice assistant**. In this mode, the model is more instruction-following. For demo, you are suggested to use `assistant_female_voice`, `assistant_male_voice`, and `assistant_default_female_voice`. Other voices may work but not as stable as the default voices.

*Please note that, `assistant_female_voice` and `assistant_male_voice` are more stable but sounds like robots, while `assistant_default_female_voice` is more human-alike but not stable, its voice often changes in multiple turns. We suggest you to try stable voices `assistant_female_voice` and `assistant_male_voice`.*

```python
ref_audio, _ = librosa.load('./assets/input_examples/assistant_female_voice.wav', sr=16000, mono=True) # or use `./assets/input_examples/assistant_male_voice.wav`
sys_prompt = model.get_sys_prompt(ref_audio=ref_audio, mode='audio_assistant', language='en') 
user_question = {'role': 'user', 'content': [librosa.load('xxx.wav', sr=16000, mono=True)[0]]} # load the user's audio question

# round one
msgs = [sys_prompt, user_question]
res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result_assistant_round_1.wav',
)

# round two
history = msgs.append({'role': 'assistant', 'content': res})
user_question = {'role': 'user', 'content': [librosa.load('xxx.wav', sr=16000, mono=True)[0]]}
msgs = history.append(user_question)
res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result_assistant_round_2.wav',
)