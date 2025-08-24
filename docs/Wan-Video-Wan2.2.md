# Wan2.2: Revolutionizing Video Generation with Advanced AI

**Wan2.2** offers cutting-edge, open-source video generation, enabling stunning cinematic-quality videos. [See the original repository](https://github.com/Wan-Video/Wan2.2) for the code and more!

[![Wan2.2 Logo](assets/logo.png)](https://github.com/Wan-Video/Wan2.2)

*   💜 [**Wan**](https://wan.video)
*   🖥️ [**GitHub**](https://github.com/Wan-Video/Wan2.2)
*   🤗 [**Hugging Face**](https://huggingface.co/Wan-AI/)
*   🤖 [**ModelScope**](https://modelscope.cn/organization/Wan-AI)
*   📑 [**Paper**](https://arxiv.org/abs/2503.20314)
*   📑 [**Blog**](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   💬 [**Discord**](https://discord.gg/AKNgpMK4Yj)
*   📕 [**使用指南(中文)**](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz)
*   📘 [**User Guide(English)**](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)
*   💬 [**WeChat(微信)**](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)

-----

## Key Features of Wan2.2

*   **Mixture-of-Experts (MoE) Architecture:** Enhances model capacity and performance without increasing computational costs.
*   **Cinematic-Level Aesthetics:** Generates videos with precise control over lighting, composition, color, and more, incorporating meticulously curated aesthetic data.
*   **Advanced Motion Generation:** Trained on a significantly expanded dataset, leading to improved generalization across motions, semantics, and aesthetics.
*   **Efficient High-Definition Hybrid TI2V:**  Offers a 5B model supporting text-to-video and image-to-video generation at 720P resolution, optimized for speed and performance on consumer-grade GPUs.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## What's New? (July 28, 2025)

*   🚀 [**HF Space**](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model.
*   💻 Wan2.2 integrated into [**ComfyUI**](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   🧩 Wan2.2's T2V, I2V and TI2V integrated into [**Diffusers**](https://github.com/huggingface/diffusers) ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   ✅ Released inference code and model weights of **Wan2.2**.

## Community Works

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides comprehensive support for Wan 2.2.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is an alternative implementation of Wan models for ComfyUI.

## Get Started with Wan2.2

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  Install dependencies:
    ```sh
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    ```

### Model Download

| Model                | Download Links                                                                                                                              | Description                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| T2V-A14B             | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE, 480P & 720P        |
| I2V-A14B             | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE, 480P & 720P       |
| TI2V-5B              | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, 720P   |

> 💡 **Note:** The TI2V-5B model supports 720P video generation at **24 FPS**.

Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Text-to-Video Generation

This repository supports the `Wan2.2-T2V-A14B` Text-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.

#### (1) Without Prompt Extension

-   Single-GPU inference
    ```sh
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

    > 💡 This command can run on a GPU with at least 80GB VRAM.
    > 💡 If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options.
-   Multi-GPU inference using FSDP + DeepSpeed Ulysses
    ```sh
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

#### (2) Using Prompt Extension

-   **Using Dashscope API for extension:**
    -   Apply for a `dashscope.api_key` in advance ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
    -   Configure the environment variable `DASH_API_KEY`. For international users, also set `DASH_API_URL`.
    -   Use the `qwen-plus` model for text-to-video tasks and `qwen-vl-max` for image-to-video tasks.
    -   Modify the model used for extension with the parameter `--prompt_extend_model`. For example:
    ```sh
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```

-   **Using a local model for extension:**
    -   Use Qwen models or other models, depending on GPU memory.
    -   Text-to-video: `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`.
    -   Image-to-video: `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`.
    -   Specify a local model path or a Hugging Face model with `--prompt_extend_model`. For example:
    ``` sh
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

### Run Image-to-Video Generation

This repository supports the `Wan2.2-I2V-A14B` Image-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.

-   Single-GPU inference
    ```sh
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

    > 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.
-   Multi-GPU inference using FSDP + DeepSpeed Ulysses
    ```sh
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
-   Image-to-Video Generation without prompt
    ```sh
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
    ```

    > 💡The model can generate videos solely from the input image. You can use prompt extension to generate prompt from the image.

### Run Text-Image-to-Video Generation

This repository supports the `Wan2.2-TI2V-5B` Text-Image-to-Video model and can support video generation at 720P resolutions.

-   Single-GPU Text-to-Video inference
    ```sh
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

    > 💡 Unlike other tasks, the 720P resolution of the Text-Image-to-Video task is `1280*704` or `704*1280`.
    > 💡 This command can run on a GPU with at least 24GB VRAM (e.g, RTX 4090 GPU).
    > 💡 If you are running on a GPU with at least 80GB VRAM, you can remove the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options.
-   Single-GPU Image-to-Video inference
    ```sh
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

    > 💡 If the image parameter is configured, it is an Image-to-Video generation; otherwise, it defaults to a Text-to-Video generation.
    > 💡 Similar to Image-to-Video, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.
-   Multi-GPU inference using FSDP + DeepSpeed Ulysses
    ```sh
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

## Computational Efficiency on Different GPUs

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

> **Note:** The table shows the computational efficiency of different Wan2.2 models on various GPUs. The results are presented in the format: **Total time (s) / peak GPU memory (GB)**.

## Introduction to Wan2.2

Wan2.2 improves upon Wan2.1 with innovations in generation quality and model capability, including:

### (1) Mixture-of-Experts (MoE) Architecture

*   Applies MoE to the video diffusion model.
*   A14B models use a two-expert design: a high-noise expert for initial layout and a low-noise expert for detail refinement.
*   Each expert model contains ~14B parameters, with 14B active per step.

<div align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</div>

<div align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</div>

### (2) Efficient High-Definition Hybrid TI2V

*   Offers a high-compression 5B model (TI2V-5B).
*   Utilizes a high-compression Wan2.2-VAE, achieving a compression ratio of 4x16x16.
*   Can generate 720P video at 24fps on a consumer-grade GPU.
*   Supports both text-to-video and image-to-video within a unified framework.

<div align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</div>

### Comparisons to SOTAs

<div align="center">
    <img src="assets/performance.png" alt="" style="width: 90%;" />
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

This project is licensed under the [Apache 2.0 License](LICENSE.txt).

## Acknowledgements

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Contact Us

Join the [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with our research and product teams!