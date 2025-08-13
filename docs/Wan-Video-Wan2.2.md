# Wan2.2: Unleashing Cinematic Video Generation with Advanced AI

**Wan2.2** is a cutting-edge, open-source video generation model that empowers you to create stunning, cinematic-quality videos from text, images, or a combination of both.  Dive into the future of video creation!  [See the original repo](https://github.com/Wan-Video/Wan2.2).

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

[**Wan Paper**](https://arxiv.org/abs/2503.20314) | [**Wan Blog**](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg) | [**Discord**](https://discord.gg/AKNgpMK4Yj) | [**User Guide (English)**](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)

---

## Key Features:

*   🎬 **Mixture-of-Experts (MoE) Architecture:**  Leverages a sophisticated MoE design to enhance model capacity without increasing computational cost, resulting in higher-quality video output.
*   🎨 **Cinematic Aesthetics Control:**  Trained with meticulously curated aesthetic data, offering unparalleled control over lighting, composition, color tones, and other elements for professional-grade video generation.
*   🚀 **Enhanced Motion and Detail:** Built on a significantly larger dataset, Wan2.2 excels at generating complex motions, richer semantics, and refined aesthetics, outperforming other open and closed-source models.
*   ⚡ **Efficient High-Definition Generation:** Open-sources a fast 5B model (TI2V-5B) capable of generating 720P videos at 24fps on consumer-grade GPUs, making it ideal for both research and industrial applications.

## Key Highlights and Updates:

*   **Latest News:** The latest integration of Wan2.2 into ComfyUI and Diffusers, and the TI2V-5B model has been released on Hugging Face.
*   **Model Availability:** Comprehensive support for Text-to-Video (T2V), Image-to-Video (I2V), and Text-Image-to-Video (TI2V) generation.
*   **Performance:** Superior video generation quality compared to leading commercial models.
*   **Efficiency:** Detailed benchmark results on computational efficiency, including multi-GPU tests.

## Demos
<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Community Works
Explore community resources:
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

## Getting Started:

### Installation:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  Install the required packages:
    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    ```

### Model Download:

Choose your model and download it from Hugging Face or ModelScope.

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |

#### Using Hugging Face CLI
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

#### Using ModelScope CLI
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Running the Models:

Detailed instructions for running text-to-video, image-to-video, and text-image-to-video generation, including multi-GPU support and prompt extension, are available in the original README.  See the original README for detailed usage instructions.

##  Introduction of Wan2.2

**Wan2.2** builds on the foundation of Wan2.1 with notable improvements in generation quality and model capability. This upgrade is driven by a series of key technical innovations, mainly including the Mixture-of-Experts (MoE) architecture, upgraded training data, and high-compression video generation.

##### (1) Mixture-of-Experts (MoE) Architecture

Wan2.2 introduces Mixture-of-Experts (MoE) architecture into the video generation diffusion model. MoE has been widely validated in large language models as an efficient approach to increase total model parameters while keeping inference cost nearly unchanged. In Wan2.2, the A14B model series adopts a two-expert design tailored to the denoising process of diffusion models: a high-noise expert for the early stages, focusing on overall layout; and a low-noise expert for the later stages, refining video details. Each expert model has about 14B parameters, resulting in a total of 27B parameters but only 14B active parameters per step, keeping inference computation and GPU memory nearly unchanged.

<div align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</div>

The transition point between the two experts is determined by the signal-to-noise ratio (SNR), a metric that decreases monotonically as the denoising step $t$ increases. At the beginning of the denoising process, $t$ is large and the noise level is high, so the SNR is at its minimum, denoted as ${SNR}_{min}$. In this stage, the high-noise expert is activated. We define a threshold step ${t}_{moe}$ corresponding to half of the ${SNR}_{min}$, and switch to the low-noise expert when $t<{t}_{moe}$.

<div align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</div>

To validate the effectiveness of the MoE architecture, four settings are compared based on their validation loss curves. The baseline **Wan2.1** model does not employ the MoE architecture. Among the MoE-based variants, the **Wan2.1 & High-Noise Expert** reuses the Wan2.1 model as the low-noise expert while uses the  Wan2.2's high-noise expert, while the **Wan2.1 & Low-Noise Expert** uses Wan2.1 as the high-noise expert and employ the Wan2.2's low-noise expert. The **Wan2.2 (MoE)** (our final version) achieves the lowest validation loss, indicating that its generated video distribution is closest to ground-truth and exhibits superior convergence.


##### (2) Efficient High-Definition Hybrid TI2V
To enable more efficient deployment, Wan2.2 also explores a high-compression design. In addition to the 27B MoE models, a 5B dense model, i.e., TI2V-5B, is released. It is supported by a high-compression Wan2.2-VAE, which achieves a $T\times H\times W$ compression ratio of $4\times16\times16$, increasing the overall compression rate to 64 while maintaining high-quality video reconstruction. With an additional patchification layer, the total compression ratio of TI2V-5B reaches $4\times32\times32$. Without specific optimization, TI2V-5B can generate a 5-second 720P video in under 9 minutes on a single consumer-grade GPU, ranking among the fastest 720P@24fps video generation models. This model also natively supports both text-to-video and image-to-video tasks within a single unified framework, covering both academic research and practical applications.


<div align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</div>



##### Comparisons to SOTAs
We compared Wan2.2 with leading closed-source commercial models on our new Wan-Bench 2.0, evaluating performance across multiple crucial dimensions. The results demonstrate that Wan2.2 achieves superior performance compared to these leading models.


<div align="center">
    <img src="assets/performance.png" alt="" style="width: 90%;" />
</div>

## Citation
If you find our work helpful, please cite us.

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

The models in this repository are licensed under the Apache 2.0 License. Please review the [LICENSE.txt](LICENSE.txt) for complete details.

## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.


## Get Involved:

Join our vibrant community and stay connected!

*   💬 [Discord](https://discord.gg/AKNgpMK4Yj)
*   💬 [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)