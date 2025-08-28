# Wan2.2: Unleash Cinematic Video Generation with Open-Source AI

**[Wan2.2](https://github.com/Wan-Video/Wan2.2)** is an advanced, open-source video generation model that empowers you to create stunning, cinematic-quality videos from text, images, or audio. This comprehensive system delivers state-of-the-art results, offering unparalleled control over aesthetics and motion.

*   **[Paper](https://arxiv.org/abs/2503.20314)** | **[Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)** | **[Discord](https://discord.gg/AKNgpMK4Yj)** | **[Hugging Face](https://huggingface.co/Wan-AI/)** | **[ModelScope](https://modelscope.cn/organization/Wan-AI)**

## Key Features of Wan2.2

*   **Mixture-of-Experts (MoE) Architecture:**  Leverages a novel MoE architecture to significantly increase model capacity while maintaining efficient computation.
*   **Cinematic Aesthetics:**  Incorporates meticulously curated aesthetic data to enable precise control over lighting, composition, color tones, and more, for professional-grade video output.
*   **Enhanced Motion Generation:** Trained on a greatly expanded dataset (+65.6% more images, +83.2% more videos) for superior performance across motion, semantics, and visual quality.
*   **Efficient High-Definition Hybrid TI2V:** Offers a high-compression 5B model supporting text-to-video and image-to-video generation at 720P and 24fps, running on consumer-grade GPUs.
*   **Supports Text-to-Video, Image-to-Video, Text-Image-to-Video, and Speech-to-Video**

## What's New?

*   **Wan2.2-S2V-14B:** Audio-driven cinematic video generation model, including inference code and a technical report. Try it on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   **HF Space:** TI2V-5B model is available on [Hugging Face](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B).
*   **ComfyUI Integration:**  Wan2.2 is integrated into [ComfyUI](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2).
*   **Diffusers Integration:** Wan2.2's T2V, I2V and TI2V integrated into [Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers).
*   **Inference Code and Model Weights:**  Released for all Wan2.2 models.

## Community Contributions

Explore community-developed resources and integrations for Wan2.2:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Offers comprehensive support for Wan 2.2, including optimization and training tools.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper):  Alternative implementation of Wan models for ComfyUI, focusing on optimizations.

## Getting Started

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  **Install Dependencies:**

    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    ```

### Model Downloads

Download the models you need from Hugging Face or ModelScope:

| Model          | Download Link                                                                                                                                    | Description                                  |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| T2V-A14B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)   🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)   | Text-to-Video MoE, 480P & 720P support        |
| I2V-A14B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)   🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)   | Image-to-Video MoE, 480P & 720P support       |
| TI2V-5B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)    | High-compression VAE, T2V+I2V, 720P support  |
| S2V-14B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)    | Speech-to-Video, 480P & 720P support          |

**Download using `huggingface-cli`:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Download using `modelscope-cli`:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Running Wan2.2

Detailed instructions for running Text-to-Video, Image-to-Video, Text-Image-to-Video, and Speech-to-Video generation are provided in the original README.  Key points and example commands are listed below.

#### Text-to-Video Generation

*   **Single-GPU (no prompt extension):**

    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

*   **Multi-GPU (FSDP + DeepSpeed):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

*   **Prompt Extension (using Dashscope API):**

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```
    *   *Or, use a local model for extension:*

    ```bash
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

#### Image-to-Video Generation

*   **Single-GPU:**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

*   **Multi-GPU:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
*   **Image-to-Video Generation without prompt:**

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
    ```

#### Text-Image-to-Video Generation

*   **Single-GPU (Text-to-Video):**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```
*   **Single-GPU (Image-to-Video):**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

*   **Multi-GPU:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

#### Speech-to-Video Generation

*   **Single-GPU:**

    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    # Without setting --num_clip, the generated video length will automatically adjust based on the input audio length
    ```

*   **Multi-GPU:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```

*   **Pose + Audio-driven Generation:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "a person is singing" --image "examples/pose.png" --audio "examples/sing.MP3" --pose_video "./examples/pose.mp4"
    ```

## Computational Efficiency

The README includes a table of computational efficiency results on various GPUs.  See the original documentation for specifics.

## Technical Details (Key Innovations)

### Mixture-of-Experts (MoE) Architecture

The MoE architecture significantly increases model capacity while keeping inference cost nearly unchanged by utilizing a two-expert design for the denoising process.

### Efficient High-Definition Hybrid TI2V

Wan2.2 provides a high-compression 5B dense model, leveraging a high-compression Wan2.2-VAE for efficient 720P video generation.

## Citation

If you use Wan2.2 in your work, please cite us:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

Wan2.2 is licensed under the Apache 2.0 License.  Please review the [LICENSE.txt](LICENSE.txt) for complete details.

## Acknowledgements

The project acknowledges the open-source contributions of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co)

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the research and product teams.