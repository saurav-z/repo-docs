# Wan2.2: Unleashing Advanced Video Generation with Open-Source AI

**Create stunning videos with unprecedented quality and efficiency using Wan2.2, the open-source video generation model that's revolutionizing AI video creation!** ([Original Repo](https://github.com/Wan-Video/Wan2.2))

[![Wan Logo](assets/logo.png)](https://github.com/Wan-Video/Wan2.2)

Wan2.2 is a significant leap forward in open-source video generation, building upon the foundation of Wan2.1. It incorporates cutting-edge innovations to deliver superior video quality, efficiency, and creative control.

**Key Features of Wan2.2:**

*   ✨ **Mixture-of-Experts (MoE) Architecture:** Leverages a sophisticated MoE architecture for enhanced model capacity while maintaining computational efficiency.
*   🎬 **Cinematic-Level Aesthetics:** Generates videos with customizable aesthetic preferences, thanks to meticulously curated aesthetic data and detailed labeling.
*   🚀 **Enhanced Complex Motion Generation:** Trained on a significantly larger dataset, Wan2.2 excels at generating complex and dynamic motions, setting a new standard for open-source video generation.
*   ⚡ **Efficient High-Definition Hybrid TI2V:**  Features a 5B model supporting both text-to-video and image-to-video generation at 720P resolution and 24fps. This model is optimized for consumer-grade GPUs.

**Key Resources:**

*   **Website:** [Wan.video](https://wan.video)
*   **GitHub:** [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
*   **Hugging Face:** [Wan-AI on Hugging Face](https://huggingface.co/Wan-AI/)
*   **ModelScope:** [Wan-AI on ModelScope](https://modelscope.cn/organization/Wan-AI)
*   **Paper:** [arXiv Paper](https://arxiv.org/abs/2503.20314)
*   **Blog:** [Wan Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   **Discord:** [Wan Discord](https://discord.gg/AKNgpMK4Yj)
*   **User Guides:** [Chinese Guide](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz) | [English Guide](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)

**Video Demos**
<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

**Latest News:**

*   **[July 28, 2025]:** TI2V-5B model available on [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B).
*   **[July 28, 2025]:** Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **[July 28, 2025]:** T2V, I2V and TI2V integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **[July 28, 2025]:** Inference code and model weights of Wan2.2 released.

**Community Works:**

*   **DiffSynth-Studio:** Provides support for Wan 2.2.
*   **Kijai's ComfyUI WanVideoWrapper:** Alternative implementation of Wan models for ComfyUI.

**(See original README for detailed Run and Installation Instructions)**

**Run Wan2.2 (Example: Text-to-Video Generation)**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**(See original README for more detailed Run Instructions)**

**Computational Efficiency on Different GPUs**

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

**Citation:**

```bibtex
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

**License:**

Wan2.2 is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for more details.

**Acknowledgements:**

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

**Contact Us:**

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with our research and product teams.