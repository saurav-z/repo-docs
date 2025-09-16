# Wan2.2: Unleashing Advanced Video Generation 

**Create stunning videos with unparalleled quality using Wan2.2, the open-source video generation model, now available at [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)!**

## Key Features

*   👍 **Mixture-of-Experts (MoE) Architecture:** Leveraging MoE for increased model capacity without compromising computational efficiency.
*   👍 **Cinematic Aesthetics:** Generate videos with precise control over lighting, composition, and color grading for a professional look.
*   👍 **Enhanced Motion & Detail:** Trained on a significantly larger dataset, delivering superior performance across motion, semantics, and aesthetics.
*   👍 **Efficient High-Definition Generation:** Featuring a 5B model with a 16x16x4 compression ratio, supporting 720P video at 24fps on consumer-grade GPUs.
*   👍 **Text, Image, and Speech-to-Video Support:** Comprehensive support for various input modalities, including text, images, and audio.

## What's New?

*   **Wan2.2-S2V-14B** - Audio-driven cinematic video generation. [Get Started](https://humanaigc.github.io/wan-s2v-webpage).
*   **HF Space** - Try out the TI2V-5B model on Hugging Face! [Link](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)
*   **ComfyUI Integration** - Utilize Wan2.2 within ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **Diffusers Integration** - T2V, I2V and TI2V models in Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **Inference Code and Model Weights** - Released for immediate use.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Quick Start

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt  # if using Speech-to-Video
```

### Model Download

Download models from Hugging Face or ModelScope:

| Model          | Download Links                                                                                                                              | Description                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| T2V-A14B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video (480P & 720P)               |
| I2V-A14B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video (480P & 720P)              |
| TI2V-5B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | Text-Image-to-Video (720P)                 |
| S2V-14B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video (480P & 720P)             |

*   **Using `huggingface-cli`:**
    ```bash
    pip install "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
    ```

*   **Using `modelscope`:**
    ```bash
    pip install modelscope
    modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
    ```

### Run Generation

*   **Text-to-Video (T2V):**

    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

    (Explore multi-GPU options using FSDP + DeepSpeed Ulysses for faster generation.)
*   **Image-to-Video (I2V):**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard..."
    ```

*   **Text-Image-to-Video (TI2V):**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
    ```

*   **Speech-to-Video (S2V):**

    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```

    (See full documentation for comprehensive generation options, including prompt extension and advanced configuration.)

## Community Works

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

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

Wan2.2 is licensed under the [Apache 2.0 License](LICENSE.txt).

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) communities!