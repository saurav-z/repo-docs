# Wan2.2: Unleash Cinematic Video Generation with Advanced AI

**Wan2.2** is an open-source, cutting-edge video generative model that empowers users to create stunning, cinematic-quality videos from text, images, and audio; [Explore the original repo](https://github.com/Wan-Video/Wan2.2).

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
</p>

-----

## Key Features

*   **Mixture-of-Experts (MoE) Architecture:** Enhances model capacity while maintaining computational efficiency by using specialized experts for different denoising stages.
*   **Cinematic Aesthetics:**  Incorporates detailed aesthetic data for precise control over lighting, composition, and color, enabling customizable video styles.
*   **Advanced Motion Generation:** Trained on a significantly expanded dataset (+65.6% more images, +83.2% more videos) leading to improved generalization across motions, semantics, and aesthetics, achieving top performance.
*   **Efficient High-Definition Generation:** Open-sources a 5B model (TI2V-5B) with a high compression ratio, supporting 720P resolution at 24fps on consumer-grade GPUs (e.g., 4090), making it fast and accessible.
*   **Speech-to-Video Capability:**  Includes support for audio-driven video generation, enabling the creation of videos from speech and accompanying image input.

## Latest Updates & News

*   **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage):** Audio-driven cinematic video generation model, including inference code, model weights, and technical report.  Try it out on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V), or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V).
*   **July 2025:** Integration with ComfyUI, Diffusers (T2V, I2V, TI2V) and Hugging Face Spaces.
*   **Text-to-Speech (TTS) support:** Integration with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation.

## Community Contributions

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides comprehensive support for Wan 2.2.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is an alternative implementation of Wan models for ComfyUI.

## To-Do List

*   [x] Wan2.2 Text-to-Video
*   [x] Wan2.2 Image-to-Video
*   [x] Wan2.2 Text-Image-to-Video
*   [ ] Wan2.2-S2V Speech-to-Video

## How to Run Wan2.2

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt # if using Speech-to-Video
```

### Model Download

Download models from Hugging Face or ModelScope using the links provided in the table below:

| Model                 | Download Links                                                                                                                              | Description                                        |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------- |
| T2V-A14B              | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE, 480P & 720P support             |
| I2V-A14B              | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE, 480P & 720P support            |
| TI2V-5B               | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-Compression VAE, T2V+I2V, 720P support        |
| S2V-14B               | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video, 480P & 720P support              |

**Note:** TI2V-5B supports 720P video generation at 24 FPS.

**Download Examples**

```bash
pip install "huggingface_hub[cli]" # or pip install modelscope
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B # Hugging Face
# OR
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B # ModelScope
```

### Running the Models

*   **Text-to-Video (T2V):**

    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

    *   Multi-GPU inference example:
        ```bash
        torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
        ```

    *   Prompt Extension (recommended for better results):  Use Dashscope API or a local model (e.g., Qwen) for detailed prompt generation.  See documentation for details on setting API keys.

*   **Image-to-Video (I2V):**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
    *   Multi-GPU example (I2V):
       ```bash
       torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
       ```

    *   I2V Generation without prompt:
       ```bash
       DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
       ```
*   **Text-Image-to-Video (TI2V):**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```
    *   Multi-GPU (TI2V):
        ```bash
        torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
        ```
    *   Single-GPU Image-to-Video inference:
       ```bash
       python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
       ```
*   **Speech-to-Video (S2V):**

    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```
    *   Multi-GPU (S2V):
        ```bash
        torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
        ```
    *   S2V with TTS:
        ```bash
        python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --enable_tts --tts_prompt_audio "examples/zero_shot_prompt.wav" --tts_prompt_text "希望你以后能够做的比我还好呦。" --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
        ```

    *   Pose + Audio generation:
        ```bash
        torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "a person is singing" --image "examples/pose.png" --audio "examples/sing.MP3" --pose_video "./examples/pose.mp4"
        ```

## Computational Efficiency

Explore performance on different GPUs:

<div align="center">
    <img src="assets/comp_effic.png" alt="Computational Efficiency" style="width: 80%;" />
</div>

## Wan2.2 Technical Deep Dive

### (1) Mixture-of-Experts (MoE) Architecture

Wan2.2 employs a MoE architecture for efficiency, utilizing specialized experts for the denoising process.
*   High-Noise Expert: Focuses on initial layout.
*   Low-Noise Expert: Refines video details.

<div align="center">
    <img src="assets/moe_arch.png" alt="MoE Architecture" style="width: 90%;" />
</div>
<div align="center">
    <img src="assets/moe_2.png" alt="MoE Transition" style="width: 90%;" />
</div>
### (2) Efficient High-Definition Hybrid TI2V

The TI2V-5B model achieves high compression and enables efficient deployment.  It uses a high-compression Wan2.2-VAE for 720P generation at 24fps, supporting both T2V and I2V tasks.

<div align="center">
    <img src="assets/vae.png" alt="VAE Compression" style="width: 80%;" />
</div>

### Comparisons to SOTAs

<div align="center">
    <img src="assets/performance.png" alt="Performance" style="width: 90%;" />
</div>

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

Licensed under the Apache 2.0 License.  Please review the [LICENSE.txt](LICENSE.txt) for full details and restrictions.

## Acknowledgements

Thanks to the contributors of SD3, Qwen, umt5-xxl, diffusers, and HuggingFace for their open research.

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for inquiries and updates.