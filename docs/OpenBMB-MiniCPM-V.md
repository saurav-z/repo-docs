<div align="center">

<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img>

**Unlock Powerful On-Device Multimodal AI: MiniCPM-V & MiniCPM-o**

</div>

MiniCPM-V and MiniCPM-o are innovative multimodal LLMs designed for efficient on-device deployment, offering cutting-edge image, video, and audio understanding.  Access the original repository here: [https://github.com/OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V).

## Key Features

*   **MiniCPM-V 4.5: The Leading Vision-Language Model**
    *   **Superior Performance:** Outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B in vision-language tasks with only 8B parameters.
    *   **Efficient Video Understanding:** Supports high-FPS and long video analysis with up to a 96x compression rate for video tokens, enabling high-FPS (up to 10FPS) and long video understanding.
    *   **Enhanced OCR and Document Processing:** Excels in OCR and document parsing, surpassing leading models in OCRBench and OmniDocBench.
    *   **Trustworthy and Multilingual:** Features trustworthy behaviors and supports over 30 languages.
    *   **Easy Deployment:** Supports llama.cpp, Ollama, vLLM, and various quantization formats (int4, GGUF, AWQ) for deployment. Also supports easy usage with LLaMA-Factory, SGLang, and quick local WebUI demo.

*   **MiniCPM-o 2.6: End-to-End Multimodal Powerhouse**
    *   **Advanced Visual Capabilities:** Achieves GPT-4o-level performance in image understanding with 8B parameters, exceeding GPT-4o-202405, Gemini 1.5 Pro, and Claude 3.5 Sonnet.
    *   **State-of-the-Art Speech Processing:** Supports bilingual real-time speech conversation with configurable voices and offers features like voice cloning, emotion control, and role-playing.
    *   **Powerful Multimodal Live Streaming:** Supports real-time video, audio streams, and interactive speech for applications like live streaming.
    *   **High Efficiency:** Offers state-of-the-art token density to improve inference speed, first-token latency, memory usage, and power consumption.

## Table of Contents

*   [MiniCPM-V 4.5](#minicpm-v-45)
    *   [Inference Efficiency](#inference-efficiency)
*   [MiniCPM-o 2.6](#minicpm-o-26)
*   [MiniCPM-V & o Cookbook](#minicpm-v--o-cookbook)
*   [Chat with Our Demo on Gradio 🤗](#chat-with-our-demo-on-gradio-)
*   [Inference](#inference)
    *   [Model Zoo](#model-zoo)
    *   [Multi-turn Conversation](#multi-turn-conversation)
    *   [Inference on Multiple GPUs](#inference-on-multiple-gpus)
    *   [Inference on Mac](#inference-on-mac)
    *   [Efficient Inference with llama.cpp, Ollama, vLLM](#efficient-inference-with-llamacpp-ollama-vllm)
*   [Fine-tuning](#fine-tuning)
*   [Awesome work using MiniCPM-V & MiniCPM-o](#awesome-work-using-minicpm-v--minicpm-o)
*   [FAQs](#faqs)
*   [Limitations](#limitations)

## MiniCPM-V 4.5

MiniCPM-V 4.5 is the latest in the MiniCPM-V series, built with 8B parameters on Qwen3-8B and SigLIP2-400M, offering significant performance gains and innovative features. Key strengths include:

*   **Leading Vision-Language Capability:** Achieves state-of-the-art performance, exceeding GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B.
*   **Efficient Video Understanding:** Enables High-FPS (up to 10FPS) video and long video understanding.
*   **Controllable Hybrid Thinking:** Features both "fast" and "deep" thinking modes for efficient usage and complex problem-solving.
*   **Enhanced OCR and Document Parsing:** Leading performance on OCRBench and OmniDocBench.
*   **Easy to Use:** Supports llama.cpp, Ollama, vLLM, quantized models, and LLaMA-Factory.

### Key Techniques

*   **Unified 3D-Resampler:** Compresses multiple video frames, enabling high-FPS video understanding without increasing LLM computational cost.
*   **Pre-training for OCR and Document Knowledge:** Improved OCR capability and knowledge from documents.
*   **Hybrid Fast/Deep Thinking:** Enhances performance and allows for efficiency and performance trade-offs.

### Evaluation

(Images and tables can be retained; these are just placeholders)

**Example Images**

<div align="center">
  <a href="https://www.youtube.com/watch?v=Cn23FujYMMU"><img src="./assets/minicpmv4_5/MiniCPM-V 4.5-8.26_img.jpeg", width=70%></a>
</div>

**Examples of Text and Image Outputs:**

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmv4_5/en_case1.png" alt="en_case1" style="margin-bottom: 5px;">
  <img src="assets/minicpmv4_5/en_case2.png" alt="en_case2" style="margin-bottom: 5px;">
  <img src="assets/minicpmv4_5/en_case3.jpeg" alt="en_case3" style="margin-bottom: 5px;">
</div>

**Example of OCR Handwriting Capabilities**

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

## MiniCPM-o 2.6

MiniCPM-o 2.6 is the latest, end-to-end model in the MiniCPM-o series, built with 8B parameters, achieving GPT-4o-level performance. Features:

*   **Leading Visual and Audio Capabilities:** Performs at the level of GPT-4o for image understanding.
*   **State-of-the-Art Speech:** Bilingual real-time speech conversation with configurable voices, outperforms GPT-4o on audio tasks.
*   **Multimodal Live Streaming:** Supports real-time interaction with continuous video and audio streams.
*   **Enhanced OCR Capabilities:** Performs best among models under 25B parameters on OCRBench.
*   **Superior Efficiency:**  Produces fewer tokens, improves inference speed, and supports multimodal live streaming on end-side devices such as iPad.
*   **Easy Deployment:** Supports llama.cpp, quantized models, and vLLM.

### Model Architecture

*   **End-to-end Omni-modal Architecture:** Connects modality encoders/decoders for fully integrated multimodal understanding.
*   **Omni-modal Live Streaming:** Uses a time-division multiplexing (TDM) mechanism for efficient omni-modality processing.
*   **Configurable Speech Modeling:** Provides flexible voice configurations and end-to-end voice cloning.

### Evaluation

(Images and tables can be retained; these are just placeholders)

**Example Image with Output**

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmo2_6/minicpmo2_6_math_intersect.png" alt="math" style="margin-bottom: 5px;">
  <img src="assets/minicpmo2_6/minicpmo2_6_diagram_train_NN.png" alt="diagram" style="margin-bottom: 5px;">
  <img src="assets/minicpmo2_6/minicpmo2_6_multi-image_bike.png" alt="bike" style="margin-bottom: 5px;">
</div>

## MiniCPM-V & o Cookbook

The MiniCPM-V & o Cookbook provides practical solutions for developers and researchers, enabling them to quickly build multimodal applications with vision, speech, and live streaming. Features:

*   **Comprehensive Documentation:** Clear and organized documentation for easy navigation.
*   **Broad User Support:** Tailored to individual users, enterprises, and researchers.
*   **Versatile Deployment Scenarios:**  Web demos, quantized deployment, and deployment on end devices.

## Chat with Our Demo on Gradio 🤗

Try out the online demo for [MiniCPM-o 2.6](https://minicpm-omni-webdemo-us.modelbest.cn/) | [MiniCPM-V 2.6](http://120.92.209.146:8887/) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2) and easily build your local demo.

## Inference

### Model Zoo

(Table of Models with memory/device requirements can be retained)

### Multi-turn Conversation

Includes code examples (can retain).

### Inference on Multiple GPUs

See [tutorial](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md) for instructions.

### Inference on Mac

(Example code can be retained)

### Efficient Inference with llama.cpp, Ollama, vLLM

(Link to forks can be retained)

## Fine-tuning

*   Simple Fine-tuning
*   With Align-Anything
*   With LLaMA-Factory
*   With the SWIFT Framework

## Awesome work using MiniCPM-V & MiniCPM-o

(List of awesome works can be retained)

## FAQs

[Link to FAQs](./docs/faqs.md)

## Limitations

*   Speech output can be flawed, with potential for noise or repetition.
*   High latency on some web demo servers.

## Model License

*   [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License.
*   MiniCPM-o/V model weights must follow [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).
*   Free academic and, with registration, free commercial use.

## Statement

Anything generated by MiniCPM-o/V models does not represent the views and positions of the model developers. We are not liable for issues arising from model usage.

## Institutions

*   THUNLP
*   ModelBest

## 🌟 Star History

(The Star History chart can be retained as an image or dynamically generated)

## Key Techniques and Other Multimodal Projects

(List of related projects can be retained)

## Citation

(Citation information can be retained)