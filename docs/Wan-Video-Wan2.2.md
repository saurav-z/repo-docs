# Wan2.2: Create Stunning Videos with Advanced AI

**Wan2.2** empowers users to generate high-quality videos with cutting-edge AI technology.  [Explore the original repository](https://github.com/Wan-Video/Wan2.2).

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

-----

**Key Features:**

*   **Mixture-of-Experts (MoE) Architecture:** Leverages MoE to increase model capacity, offering enhanced performance without increased computational cost.
*   **Cinematic Aesthetics:** Includes data specifically designed for cinematic style generation, providing control over lighting, composition, and color.
*   **Advanced Motion Generation:** Trained on significantly larger datasets, resulting in superior motion, semantic, and aesthetic generalization.
*   **Efficient High-Definition (HD) Hybrid TI2V:**  Provides a 5B model for text-to-video (T2V) and image-to-video (I2V) generation at 720P and 24fps, even on consumer-grade GPUs.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Latest News

*   **Aug 26, 2025**: Released **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage)**, an audio-driven cinematic video generation model.
*   **Jul 28, 2025**: Released  HF space, ComfyUI, and Diffusers integrations for Wan2.2.

## Community Works

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Todo List
(Same as original README)

## Run Wan2.2

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

**T2V-A14B** Text-to-Video, supports 480P & 720P
*   🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
*   🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)

**I2V-A14B** Image-to-Video, supports 480P & 720P
*   🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
*   🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)

**TI2V-5B** High-compression VAE, T2V+I2V, supports 720P
*   🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
*   🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)

**S2V-14B** Speech-to-Video, supports 480P & 720P
*   🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
*   🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)

Download with huggingface-cli:
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Download with modelscope-cli:
```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Text-to-Video Generation
(Same as original README)

### Run Image-to-Video Generation
(Same as original README)

### Run Text-Image-to-Video Generation
(Same as original README)

### Run Speech-to-Video Generation
(Same as original README)

## Computational Efficiency on Different GPUs
(Same as original README)

## Introduction of Wan2.2
(Same as original README)

## Citation
(Same as original README)

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License Agreement
(Same as original README)

## Acknowledgements
(Same as original README)

## Contact Us
(Same as original README)