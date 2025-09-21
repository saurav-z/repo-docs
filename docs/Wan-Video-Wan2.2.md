# Wan2.2: Unleashing Advanced Video Generation with Open-Source Innovation

**Wan2.2** is a cutting-edge open-source video generation model, offering state-of-the-art performance and innovative features. Build upon [the original Wan2.2 repository](https://github.com/Wan-Video/Wan2.2), Wan2.2 empowers users to create high-quality videos from text, images, audio, and more.

**Key Features:**

*   ✅ **Mixture-of-Experts (MoE) Architecture:**  Leverages a MoE design for enhanced model capacity and efficiency, resulting in superior video quality.
*   ✅ **Cinematic Aesthetics:**  Generates videos with meticulous attention to aesthetic details, including lighting, composition, and color grading, for professional-grade results.
*   ✅ **Complex Motion Generation:** Trained on a massive dataset, Wan2.2 excels at creating dynamic and realistic motion in videos.
*   ✅ **Efficient High-Definition (720P) Hybrid TI2V:** Open-sources a fast and efficient 5B model, enabling text-to-video and image-to-video generation at 720P resolution on consumer-grade GPUs.
*   ✅ **Versatile Generation Modes:** Supports Text-to-Video (T2V), Image-to-Video (I2V), Text-Image-to-Video (TI2V), Speech-to-Video (S2V), and character animation/replacement.

## Table of Contents

*   [Latest News](#latest-news)
*   [Video Demos](#video-demos)
*   [Community Works](#community-works)
*   [Getting Started](#run-wan2.2)
    *   [Installation](#installation)
    *   [Model Download](#model-download)
    *   [Running Text-to-Video Generation](#run-text-to-video-generation)
    *   [Running Image-to-Video Generation](#run-image-to-video-generation)
    *   [Running Text-Image-to-Video Generation](#run-text-image-to-video-generation)
    *   [Running Speech-to-Video Generation](#run-speech-to-video-generation)
    *   [Running Wan-Animate](#run-wan-animate)
*   [Computational Efficiency](#computational-efficiency-on-different-gpus)
*   [Model Details](#introduction-of-wan2.2)
*   [Citation](#citation)
*   [License](#license-agreement)
*   [Acknowledgments](#acknowledgements)
*   [Contact](#contact-us)

## Latest News

*   **Sep 19, 2025:** Released **[Wan2.2-Animate-14B](https://humanaigc.github.io/wan-animate)** for character animation and replacement.
*   **Aug 26, 2025:** Introduced **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage)**, an audio-driven cinematic video generation model.
*   **Jul 28, 2025:** Released TI2V-5B model on Hugging Face Spaces, ComfyUI and Diffusers integrations, and inference code & model weights of Wan2.2.
*   **Sep 5, 2025:** Added text-to-speech support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation.

## Video Demos

[Include video demo here if possible. If not, replace with a descriptive text.]

## Community Works

[List of community projects that use Wan2.2]

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Run Wan2.2

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt  # If using Speech-to-Video
```

### Model Download

[Table of Models and Download Links]

### Run Text-to-Video Generation

[Instructions for running T2V generation, including single-GPU and multi-GPU examples, and information on prompt extension.]

### Run Image-to-Video Generation

[Instructions for running I2V generation, including single-GPU and multi-GPU examples, and information on prompt extension.]

### Run Text-Image-to-Video Generation

[Instructions for running TI2V generation, including single-GPU and multi-GPU examples.]

### Run Speech-to-Video Generation

[Instructions for running S2V generation, including single-GPU and multi-GPU examples.]

### Run Wan-Animate

[Instructions for running Wan-Animate, including preprocessing steps and inference examples.]

## Computational Efficiency on Different GPUs

[Table or Image of GPU performance results.]

## Model Details

[Detailed overview of the Wan2.2 architecture, MoE design, and high-compression TI2V model, along with performance comparisons.]

## Citation

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License Agreement

[Link to LICENSE.txt and a brief summary of the license terms.]

## Acknowledgments

[List of acknowledgements]

## Contact Us

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with our research and product teams.