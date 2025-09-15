# Wan2.2: Generate Stunning Videos with Advanced AI

**Unleash the power of advanced AI with Wan2.2, an open-source video generation model that sets a new standard for quality and efficiency.** ([Original Repo](https://github.com/Wan-Video/Wan2.2))

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

**Key Features:**

*   **Mixture-of-Experts (MoE) Architecture:** Achieve higher model capacity and improved detail through specialized expert models, all while maintaining the same computational cost.
*   **Cinematic-Level Aesthetics:** Generate videos with precise control over lighting, composition, and color, thanks to meticulously curated aesthetic data.
*   **Enhanced Motion Generation:** Create more dynamic and complex motion sequences with a significantly expanded training dataset (+65.6% images, +83.2% videos) compared to Wan2.1.
*   **Efficient High-Definition Hybrid TI2V:** Experience fast 720P@24fps video generation with our open-sourced 5B model, ideal for both industrial and academic applications.

## Quick Start

*   **Installation:**

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    pip install -r requirements.txt
    pip install -r requirements_s2v.txt  # if using Speech-to-Video
    ```

*   **Model Download:**  Choose from the following models and download using Hugging Face or ModelScope CLI. Download links provided in the "Run Wan2.2" section below.

    | Model             | Description                                     |
    | :---------------- | :---------------------------------------------- |
    | T2V-A14B          | Text-to-Video                                   |
    | I2V-A14B          | Image-to-Video                                  |
    | TI2V-5B           | Text-Image-to-Video (720P)                      |
    | S2V-14B           | Speech-to-Video                                 |

*   **Generation Examples:**  See below for example commands to generate videos from text, images, and speech!

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Run Wan2.2

Detailed instructions and example code snippets for running the models are provided below.  This covers Text-to-Video, Image-to-Video, Text-Image-to-Video and Speech-to-Video generation.

#### Model Download

Download your desired models using either the Hugging Face Hub or ModelScope:

```bash
# Hugging Face (replace MODEL_NAME)
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/MODEL_NAME --local-dir ./MODEL_NAME

# ModelScope (replace MODEL_NAME)
pip install modelscope
modelscope download Wan-AI/MODEL_NAME --local_dir ./MODEL_NAME
```

Replace `MODEL_NAME` with one of the following:
*   Wan2.2-T2V-A14B
*   Wan2.2-I2V-A14B
*   Wan2.2-TI2V-5B
*   Wan2.2-S2V-14B

#### Run Text-to-Video Generation

*   **Single-GPU (480P & 720P):**

    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

*   **Prompt Extension (Dashscope API):**  Requires a Dashscope API key.

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```

*   **Prompt Extension (Local Qwen Model):**

    ```bash
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

#### Run Image-to-Video Generation

*   **Single-GPU (480P & 720P):**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

*   **Image-to-Video Generation without prompt (with prompt extension):**

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
    ```

#### Run Text-Image-to-Video Generation

*   **Single-GPU (720P):**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

*   **Single-GPU Image-to-Video (720P):**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

*   **Multi-GPU (FSDP + DeepSpeed Ulysses) (720P):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

#### Run Speech-to-Video Generation

*   **Single-GPU:**

    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```

    ```bash
    # With CosyVoice text-to-speech:
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --enable_tts --tts_prompt_audio "examples/zero_shot_prompt.wav" --tts_prompt_text "希望你以后能够做的比我还好呦。" --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    ```

*   **Multi-GPU:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```

*   **Pose + Audio driven generation:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "a person is singing" --image "examples/pose.png" --audio "examples/sing.MP3" --pose_video "./examples/pose.mp4"
    ```

## Latest News

*   **Aug 26, 2025:** Introducing **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage)**, an audio-driven cinematic video generation model, with [inference code](#run-speech-to-video-generation), [model weights](#model-download), and [technical report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)!
*   **Try it out:** [wan.video](https://wan.video/),  [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   **July 28, 2025:**  [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)  using the TI2V-5B model.
*   **July 28, 2025:**  Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **July 28, 2025:** Wan2.2's T2V, I2V and TI2V integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:** Inference code and model weights of **Wan2.2** released.
*   **Sep 5, 2025:** Text-to-speech support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation.

## Community Works

Explore projects built upon [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) or [**Wan2.2**](https://github.com/Wan-Video/Wan2.2):

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Technical Details

### Mixture-of-Experts (MoE) Architecture

Wan2.2 utilizes a Mixture-of-Experts (MoE) architecture to enhance model capacity and improve generation quality. This design enables larger models without a significant increase in inference costs.  The A14B models incorporate two expert models optimized for different stages of the denoising process.

### Efficient High-Definition Hybrid TI2V

A 5B dense model, TI2V-5B, with a high-compression Wan2.2-VAE supports high-quality 720P video generation while remaining fast and accessible.  This model supports both text-to-video and image-to-video generation.

### Performance

Wan2.2 achieves superior performance compared to leading commercial models, as demonstrated in the performance comparison chart provided in the original README.

## Citation

If you find our work helpful, please cite us using:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

Wan2.2 is licensed under the Apache 2.0 License.

## Acknowledgements

We thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories for their open research.

## Contact Us

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the research and product teams.