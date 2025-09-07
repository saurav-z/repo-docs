# Wan2.2: Unleash Cinematic Video Generation with Open-Source AI

**Wan2.2** is a groundbreaking open-source video generation model, offering state-of-the-art results and innovative features.  Explore the future of video creation with this powerful, advanced model! ([Original Repo](https://github.com/Wan-Video/Wan2.2))

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

-----

**Key Features of Wan2.2:**

*   👍 **Mixture-of-Experts (MoE) Architecture:**  Enhanced model capacity with specialized experts for different stages of video denoising.
*   👍 **Cinematic Aesthetics:**  Precise control over style through meticulously curated aesthetic data, including lighting, composition, and color.
*   👍 **Advanced Motion Generation:**  Trained on significantly expanded datasets for improved generalization across motion, semantics, and aesthetics.
*   👍 **Efficient High-Definition TI2V:**  Open-sourced 5B model offering fast 720P video generation at 24fps, running on consumer-grade GPUs.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

##  Latest Updates:

*   **[Aug 26, 2025]:** Introducing **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage)**, an audio-driven cinematic video generation model, including [inference code](#run-speech-to-video-generation), [model weights](#model-download), and [technical report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)! Now you can try it on [wan.video](https://wan.video/),  [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   **[Jul 28, 2025]:**  HF Space for TI2V-5B Model: [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)
*   **[Jul 28, 2025]:**  ComfyUI Integration:  Wan2.2 is now integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **[Jul 28, 2025]:**  Diffusers Integration:  Wan2.2's T2V, I2V, and TI2V are integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **[Jul 28, 2025]:**  Inference Code and Model Weights Released.
*   **[Sep 5, 2025]:** Text-to-Speech Integration: Text-to-speech synthesis support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation task.

## Community Contributions

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides comprehensive support for Wan 2.2.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is an alternative implementation of Wan models for ComfyUI.

## Roadmap / Todo List

*   Wan2.2 Text-to-Video
    *   [x] Multi-GPU Inference code of the A14B and 14B models
    *   [x] Checkpoints of the A14B and 14B models
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   Wan2.2 Image-to-Video
    *   [x] Multi-GPU Inference code of the A14B model
    *   [x] Checkpoints of the A14B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   Wan2.2 Text-Image-to-Video
    *   [x] Multi-GPU Inference code of the 5B model
    *   [x] Checkpoints of the 5B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   Wan2.2-S2V Speech-to-Video
    *   [x] Inference code of Wan2.2-S2V
    *   [x] Checkpoints of Wan2.2-S2V-14B
    *   [ ] ComfyUI integration
    *   [ ] Diffusers integration

## Run Wan2.2: Quickstart

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
# If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
pip install -r requirements_s2v.txt
```

### Model Download

Download the necessary models from Hugging Face or ModelScope:

| Model              | Download Link                                                                                                                             | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |

> **Note:** The TI2V-5B model supports 720P video generation at **24 FPS**.

#### Using Hugging Face CLI

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

#### Using ModelScope CLI

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Generate Videos:

#### Text-to-Video Generation

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

*   Supports 480P and 720P resolutions.
*   Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` for reduced GPU memory usage.
*   Multi-GPU with FSDP + DeepSpeed Ulysses:
    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

#### Image-to-Video Generation

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   Supports 480P and 720P resolutions.
*   Multi-GPU with FSDP + DeepSpeed Ulysses (example in original README).

#### Text-Image-to-Video Generation

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

*   Supports 720P resolution (1280\*704 or 704\*1280).

#### Speech-to-Video Generation

```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
# Generate video length automatically adjusts
```
*   Supports 480P and 720P resolutions.
*   Multi-GPU with FSDP + DeepSpeed Ulysses (example in original README).
*   Pose + Audio driven generation (example in original README).

### Prompt Extension:

Use the Dashscope API (requires API key and environment variables) or a local model for prompt enhancement to enrich video details.

## Computational Efficiency

See the original README for a table comparing the speed and memory usage of different models on various GPUs.

## Introduction of Wan2.2

(Summary of original "Introduction" section - key points are in the Key Features section)

## Citation

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

Licensed under the Apache 2.0 License.  Refer to [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

(Original Acknowledgements Section)

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the research and product teams.
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  Uses "Unleash Cinematic Video Generation" for a compelling opening.
*   **Concise Summary:**  Presents key features upfront.
*   **Structured Headings:**  Uses clear headings (Installation, Model Download, Run, etc.) for readability and SEO.
*   **Bulleted Lists:**  Employs bullet points for key features, making the information scannable.
*   **Keyword Integration:**  Includes relevant keywords like "video generation," "open-source," "AI," "cinematic," etc.
*   **Emphasis on Value:** Highlights the benefits of using the model.
*   **Modelscope Links:**  Modelscope and HF links included to help with ranking.
*   **Code Highlighting:**  Uses code blocks for commands, enhancing clarity.
*   **Simplified Instructions:**  Provides a streamlined quickstart guide.
*   **Concise Intro Section:** Provides a short version of the original description.
*   **Community Section**:  Includes the community section so people can easily get to their work.